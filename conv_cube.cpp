#include <torch/all.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include <map>
#include <tuple>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul/matmul_tiling.h"
#include "platform/platform_ascendc.h"

namespace thin { aclrtStream stream(); void sync(); void select(c10::Device); }

namespace image_conv {
at::Tensor conv(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
                int64_t, int64_t, bool, bool);
}

namespace conv_cube {
using namespace AscendC;
constexpr int cores = 8;
constexpr int tileN = 256;

__aicore__ inline void input_ready()
{
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}
__aicore__ inline void output_ready()
{
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}
__aicore__ inline void output_done()
{
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}
__aicore__ inline void split_float(LocalTensor<half> hi, LocalTensor<half> lo,
                                    LocalTensor<float> src, LocalTensor<float> tmp, int count)
{
    for(int offset=0;offset<count;offset+=64*255) {
        int chunk=count-offset<64*255?count-offset:64*255;
        Cast(hi[offset], src[offset], RoundMode::CAST_NONE, 64, chunk/64, {1,1,4,8});
        PipeBarrier<PIPE_V>();
        Cast(tmp[offset], hi[offset], RoundMode::CAST_NONE, 64, chunk/64, {1,1,8,4});
        PipeBarrier<PIPE_V>();
        Sub(tmp[offset], src[offset], tmp[offset], chunk);
        PipeBarrier<PIPE_V>();
        Cast(lo[offset], tmp[offset], RoundMode::CAST_NONE, 64, chunk/64, {1,1,4,8});
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void zero_column(LocalTensor<half> dst, int height, int width, int col)
{
    int group = 1;
    while ((width * group) % 16 != 0) ++group;
    for (int row = 0; row < group && row < height; ++row) {
        int index = row * width + col;
        int base = index / 16 * 16;
        uint64_t mask[2] = {1ULL << (index-base), 0};
        Duplicate(dst[base], (half)0, mask, (height-row+group-1)/group, 1, width*group/16);
    }
}

template<bool nz>
__aicore__ inline void store_patch(GlobalTensor<half> output, LocalTensor<half> patch,
                                    int row, int spatial, int k)
{
    if constexpr (nz) {
        DataCopy(output[row*16], patch, DataCopyParams{(uint16_t)(spatial/16),1,0,(uint16_t)(k-1)});
    } else {
        DataCopy(output[row*spatial], patch, spatial);
    }
}

template<class T, bool nz>
__global__ __aicore__ void im2col_kernel(GM_ADDR x, GM_ADDR hi, GM_ADDR lo,
                                         int channels, int height, int width, int kp)
{
    TPipe pipe;
    int spatial=height*width, padded=(spatial+63)/64*64;
    int guard=(width+31)/16*16;
    TBuf<TPosition::VECCALC> hiSrcBuf,loSrcBuf,hiDstBuf,loDstBuf,srcBuf,tmpBuf;
    pipe.InitBuffer(hiSrcBuf,(padded+2*guard)*2);
    pipe.InitBuffer(hiDstBuf,padded*2);
    if constexpr (std::is_same<T,float>::value) {
        pipe.InitBuffer(loSrcBuf,(padded+2*guard)*2);
        pipe.InitBuffer(loDstBuf,padded*2);
        pipe.InitBuffer(srcBuf,padded*4);
        pipe.InitBuffer(tmpBuf,padded*4);
    }
    auto hiSrc=hiSrcBuf.Get<half>(),hiDst=hiDstBuf.Get<half>();
    GlobalTensor<T> input;
    GlobalTensor<half> high,low;
    input.SetGlobalBuffer((__gm__ T*)x);
    high.SetGlobalBuffer((__gm__ half*)hi);
    low.SetGlobalBuffer((__gm__ half*)lo);
    Duplicate(hiSrc,(half)0,padded+2*guard);
    if constexpr (std::is_same<T,float>::value) {
        Duplicate(loSrcBuf.Get<half>(),(half)0,padded+2*guard);
        Duplicate(srcBuf.Get<float>(),0.0f,padded);
    }
    PipeBarrier<PIPE_ALL>();
    for (int c=GetBlockIdx();c<channels;c+=GetBlockNum()) {
        if constexpr (std::is_same<T,float>::value) {
            auto src=srcBuf.Get<float>();
            DataCopy(src,input[c*spatial],spatial);
            input_ready();
            split_float(hiSrc[guard],loSrcBuf.Get<half>()[guard],src,tmpBuf.Get<float>(),padded);
        } else {
            DataCopy(hiSrc[guard],input[c*spatial],spatial);
            input_ready();
        }
        for (int p=0;p<9;++p) {
            int dy=p/3-1,dx=p%3-1;
            Adds(hiDst,hiSrc[guard+dy*width+dx],(half)0,padded);
            if constexpr (std::is_same<T,float>::value) {
                Adds(loDstBuf.Get<half>(),loSrcBuf.Get<half>()[guard+dy*width+dx],(half)0,padded);
            }
            PipeBarrier<PIPE_V>();
            if(dx!=0) {
                zero_column(hiDst,height,width,dx<0?0:width-1);
                if constexpr (std::is_same<T,float>::value) zero_column(loDstBuf.Get<half>(),height,width,dx<0?0:width-1);
                PipeBarrier<PIPE_V>();
            }
            output_ready();
            store_patch<nz>(high,hiDst,c*9+p,spatial,kp);
            if constexpr (std::is_same<T,float>::value) store_patch<nz>(low,loDstBuf.Get<half>(),c*9+p,spatial,kp);
            output_done();
        }
    }
    Duplicate(hiDst,(half)0,padded);
    output_ready();
    for(int row=channels*9+GetBlockIdx();row<kp;row+=GetBlockNum()) {
        store_patch<nz>(high,hiDst,row,spatial,kp);
        if constexpr(std::is_same<T,float>::value) store_patch<nz>(low,hiDst,row,spatial,kp);
    }
    output_done();
}

__global__ __aicore__ void split_weights_kernel(GM_ADDR w, GM_ADDR hi, GM_ADDR lo, int total)
{
    TPipe pipe;
    constexpr int tile=4096;
    TBuf<TPosition::VECCALC> srcBuf,hiBuf,loBuf,tmpBuf;
    pipe.InitBuffer(srcBuf,tile*4);
    pipe.InitBuffer(hiBuf,tile*2);
    pipe.InitBuffer(loBuf,tile*2);
    pipe.InitBuffer(tmpBuf,tile*4);
    auto src=srcBuf.Get<float>(),tmp=tmpBuf.Get<float>();
    auto high=hiBuf.Get<half>(),low=loBuf.Get<half>();
    GlobalTensor<float> input;
    GlobalTensor<half> h,l;
    input.SetGlobalBuffer((__gm__ float*)w);
    h.SetGlobalBuffer((__gm__ half*)hi);
    l.SetGlobalBuffer((__gm__ half*)lo);
    for(int offset=GetBlockIdx()*tile;offset<total;offset+=GetBlockNum()*tile) {
        int count=total-offset<tile?total-offset:tile;
        DataCopy(src,input[offset],count);
        input_ready();
        split_float(high,low,src,tmp,(count+63)/64*64);
        output_ready();
        DataCopy(h[offset],high,count);
        DataCopy(l[offset],low,count);
        output_done();
    }
}

template<bool compensated,bool nz,class Out>
__global__ __aicore__ void conv_mm_kernel(GM_ADDR wh,GM_ADDR wl,GM_ADDR bh,GM_ADDR bl,GM_ADDR y,
                                          TCubeTiling tiling,int m,int n,int k,int tileM)
{
    using A=MatmulType<TPosition::GM,CubeFormat::ND,half>;
    using B=MatmulType<TPosition::GM,nz?CubeFormat::NZ:CubeFormat::ND,half>;
    using C=MatmulType<TPosition::GM,CubeFormat::ND,Out>;
    TPipe pipe;
    Matmul<A,B,C> mm;
    TBuf<TPosition::VECCALC> workspace;
    pipe.InitBuffer(workspace,tiling.transLength*4);
    mm.Init(&tiling,&pipe);
    mm.SetLocalWorkspace(workspace.Get<uint8_t>());
    GlobalTensor<half> ah,al,high,low;
    GlobalTensor<Out> out;
    ah.SetGlobalBuffer((__gm__ half*)wh);
    al.SetGlobalBuffer((__gm__ half*)wl);
    high.SetGlobalBuffer((__gm__ half*)bh);
    low.SetGlobalBuffer((__gm__ half*)bl);
    out.SetGlobalBuffer((__gm__ Out*)y);
    int tilesN=(n+tileN-1)/tileN,tilesM=(m+tileM-1)/tileM;
    for(int tile=GetBlockIdx();tile<tilesM*tilesN;tile+=GetBlockNum()) {
        int row=tile/tilesN*tileM,col=tile%tilesN*tileN;
        int rows=m-row<tileM?m-row:tileM,cols=n-col<tileN?n-col:tileN;
        int bOffset=nz?col*k:col;
        mm.SetTail(rows,cols,k);
        mm.SetTensorA(ah[row*k]);
        mm.SetTensorB(high[bOffset]);
        mm.IterateAll(out[row*n+col]);
        mm.End();
        if constexpr(compensated) {
            mm.SetTail(rows,cols,k);
            mm.SetTensorA(al[row*k]);
            mm.SetTensorB(high[bOffset]);
            mm.IterateAll(out[row*n+col],1);
            mm.End();
            mm.SetTail(rows,cols,k);
            mm.SetTensorA(ah[row*k]);
            mm.SetTensorB(low[bOffset]);
            mm.IterateAll(out[row*n+col],1);
            mm.End();
        }
    }
}

TCubeTiling make_tiling(int m,int n,int k,bool nz)
{
    using Key=std::tuple<int,int,int,bool>;
    static thread_local std::map<Key,TCubeTiling> cache;
    Key key{m,n,k,nz};
    auto found=cache.find(key);
    if(found!=cache.end()) return found->second;
    auto p=platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MatmulApiTiling t(*p);
    t.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT16);
    t.SetBType(matmul_tiling::TPosition::GM,nz?matmul_tiling::CubeFormat::NZ:matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT16);
    t.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
    int tm=m<64?m:64;
    t.SetShape(tm,tileN,k);
    t.SetOrgShape(m,n,k);
    t.SetBias(false);
    t.SetFixSplit(tm,tileN,-1);
    t.SetBufferSpace(-1,-1,-1);
    TCubeTiling td;
    TORCH_CHECK(t.GetTiling(td)==0,"310P Matmul tiling failed");
    cache.emplace(key,td);
    return td;
}

template<class T, bool nz>
__global__ __aicore__ void tiled_im2col_kernel(GM_ADDR x, GM_ADDR hi, GM_ADDR lo,
    int channels,int height,int width,int kernel,int stride,int pad,int oh,int ow,int np,int kp)
{
    constexpr int tile=2048;
    TPipe pipe;
    int capacity=(tile*stride*stride+(kernel+stride)*width+191)/64*64;
    int gatherStride=(capacity/2+63)/64*64;
    int gatherCapacity=2*gatherStride>tile*3+128?2*gatherStride:tile*3+128;
    TBuf<TPosition::VECCALC> rawBuf,hiBuf,loBuf,tmpBuf,dhBuf,dlBuf,gatherBuf;
    pipe.InitBuffer(rawBuf,capacity*sizeof(T));
    pipe.InitBuffer(hiBuf,(capacity+128)*2);
    pipe.InitBuffer(loBuf,(capacity+128)*2);
    pipe.InitBuffer(tmpBuf,capacity*4);
    pipe.InitBuffer(dhBuf,tile*2);
    pipe.InitBuffer(dlBuf,tile*2);
    pipe.InitBuffer(gatherBuf,gatherCapacity*2);
    auto raw=rawBuf.Get<T>();
    auto high=hiBuf.Get<half>(),low=loBuf.Get<half>();
    auto dh=dhBuf.Get<half>(),dl=dlBuf.Get<half>(),gather=gatherBuf.Get<half>();
    GlobalTensor<T> input; input.SetGlobalBuffer((__gm__ T*)x);
    GlobalTensor<half> outH,outL;
    outH.SetGlobalBuffer((__gm__ half*)hi); outL.SetGlobalBuffer((__gm__ half*)lo);
    int n=oh*ow,tiles=(np+tile-1)/tile,kk=kernel*kernel;
    for(int task=GetBlockIdx();task<channels*tiles;task+=GetBlockNum()) {
        int c=task/tiles,base=task%tiles*tile;
        int count=n-base<tile?n-base:tile;
        int allocated=np-base<tile?np-base:tile;
        if(count<=0) continue;
        int firstY=base/ow*stride-pad;
        int lastY=(base+count-1)/ow*stride+kernel-1-pad;
        int validFirst=firstY<0?0:firstY;
        int validLast=lastY>=height?height-1:lastY;
        int start=c*height*width+validFirst*width;
        constexpr int block=32/sizeof(T);
        int aligned=start/block*block,skew=start-aligned;
        int loaded=(skew+(validLast-validFirst+1)*width+block-1)/block*block;
        int split=(loaded+63)/64*64;
        Duplicate(raw,(T)0,split);
        Duplicate(high,(half)0,capacity+128);
        if constexpr(std::is_same<T,float>::value) Duplicate(low,(half)0,capacity+128);
        PipeBarrier<PIPE_ALL>();
        DataCopy(raw,input[aligned],loaded);
        input_ready();
        if constexpr(std::is_same<T,float>::value) {
            split_float(high[64],low[64],raw,tmpBuf.Get<float>(),split);
        } else {
            Adds(high[64],raw,(half)0,split);
            PipeBarrier<PIPE_V>();
        }
        for(int p=0;p<kk;++p) {
            Duplicate(dh,(half)0,tile);
            if constexpr(std::is_same<T,float>::value) Duplicate(dl,(half)0,tile);
            PipeBarrier<PIPE_V>();
            if(stride==2 && width%2==0) {
                int source=64+skew+p%kernel-pad;
                int elements=(validLast-validFirst+1)*width;
                auto shifted=tmpBuf.Get<half>();
                uint64_t selected=0;
                Adds(shifted,high[source],(half)0,elements);
                PipeBarrier<PIPE_V>();
                GatherMask(gather,shifted,(uint8_t)1,true,elements,{1,1,8,0},selected);
                PipeBarrier<PIPE_V>();
                if constexpr(std::is_same<T,float>::value) {
                    Adds(shifted,low[source],(half)0,elements);
                    PipeBarrier<PIPE_V>();
                    GatherMask(gather[gatherStride],shifted,(uint8_t)1,true,elements,{1,1,8,0},selected);
                    PipeBarrier<PIPE_V>();
                }
            }
            for(int offset=0;offset<count;) {
                int pos=base+offset,y=pos/ow,ox=pos%ow;
                int len=ow-ox<count-offset?ow-ox:count-offset;
                int iy=y*stride+p/kernel-pad;
                int ix=ox*stride+p%kernel-pad;
                if(iy>=0 && iy<height) {
                    int source=64+skew+(iy-validFirst)*width+ix;
                    if(stride==2 && width%2==0) {
                        int decimated=(iy-validFirst)*(width/2)+ox;
                        Adds(dh[offset],gather[decimated],(half)0,len);
                        if constexpr(std::is_same<T,float>::value) Adds(dl[offset],gather[gatherStride+decimated],(half)0,len);
                    } else if(stride==1) {
                        Adds(dh[offset],high[source],(half)0,len);
                        if constexpr(std::is_same<T,float>::value) Adds(dl[offset],low[source],(half)0,len);
                    } else {
                        uint64_t selected=0;
                        // GatherMask needs a 32-byte aligned source, unlike the vector shift.
                        Adds(gather,high[source],(half)0,len*2);
                        PipeBarrier<PIPE_V>();
                        GatherMask(gather[4096],gather,(uint8_t)1,true,len*2,{1,1,8,0},selected);
                        PipeBarrier<PIPE_V>();
                        Adds(dh[offset],gather[4096],(half)0,len);
                        PipeBarrier<PIPE_V>();
                        if constexpr(std::is_same<T,float>::value) {
                            Adds(gather,low[source],(half)0,len*2);
                            PipeBarrier<PIPE_V>();
                            GatherMask(gather[4096],gather,(uint8_t)1,true,len*2,{1,1,8,0},selected);
                            PipeBarrier<PIPE_V>();
                            Adds(dl[offset],gather[4096],(half)0,len);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    if(ix<0) {
                        uint64_t mask[2]={1ULL<<((offset)%16),0};
                        Duplicate(dh[offset/16*16],(half)0,mask,1,1,8);
                        if constexpr(std::is_same<T,float>::value) Duplicate(dl[offset/16*16],(half)0,mask,1,1,8);
                    }
                    if(ix+(len-1)*stride>=width) {
                        int end=offset+len-1;
                        uint64_t mask[2]={1ULL<<(end%16),0};
                        Duplicate(dh[end/16*16],(half)0,mask,1,1,8);
                        if constexpr(std::is_same<T,float>::value) Duplicate(dl[end/16*16],(half)0,mask,1,1,8);
                    }
                    PipeBarrier<PIPE_V>();
                }
                offset+=len;
            }
            output_ready();
            int row=c*kk+p;
            if constexpr(nz) {
                DataCopy(outH[row*16+base*kp],dh,DataCopyParams{(uint16_t)(allocated/16),1,0,(uint16_t)(kp-1)});
                if constexpr(std::is_same<T,float>::value) DataCopy(outL[row*16+base*kp],dl,DataCopyParams{(uint16_t)(allocated/16),1,0,(uint16_t)(kp-1)});
            } else {
                DataCopy(outH[row*np+base],dh,allocated);
                if constexpr(std::is_same<T,float>::value) DataCopy(outL[row*np+base],dl,allocated);
            }
            output_done();
        }
    }
    Duplicate(dh,(half)0,tile);
    output_ready();
    for(int row=channels*kk+GetBlockIdx();row<kp;row+=GetBlockNum()) {
        for(int base=0;base<np;base+=tile) {
            int count=np-base<tile?np-base:tile;
            if constexpr(nz) {
                DataCopy(outH[row*16+base*kp],dh,DataCopyParams{(uint16_t)(count/16),1,0,(uint16_t)(kp-1)});
                if constexpr(std::is_same<T,float>::value) DataCopy(outL[row*16+base*kp],dh,DataCopyParams{(uint16_t)(count/16),1,0,(uint16_t)(kp-1)});
            } else {
                DataCopy(outH[row*np+base],dh,count);
                if constexpr(std::is_same<T,float>::value) DataCopy(outL[row*np+base],dh,count);
            }
        }
    }
    output_done();
}

template<class T>
__global__ __aicore__ void pack_weights_kernel(GM_ADDR w,GM_ADDR hi,GM_ADDR lo,int o,int kr,int kp,int mp)
{
    constexpr int tile=4096;
    constexpr int block=32/sizeof(T);
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf,hiBuf,loBuf,tmpBuf,dhBuf,dlBuf;
    pipe.InitBuffer(rawBuf,(tile+64)*sizeof(T));
    pipe.InitBuffer(hiBuf,(tile+64)*2); pipe.InitBuffer(loBuf,(tile+64)*2);
    pipe.InitBuffer(tmpBuf,(tile+64)*4);
    pipe.InitBuffer(dhBuf,tile*2); pipe.InitBuffer(dlBuf,tile*2);
    auto raw=rawBuf.Get<T>(); auto h=hiBuf.Get<half>(),l=loBuf.Get<half>();
    auto dh=dhBuf.Get<half>(),dl=dlBuf.Get<half>();
    GlobalTensor<T> input; input.SetGlobalBuffer((__gm__ T*)w);
    GlobalTensor<half> outH,outL;
    outH.SetGlobalBuffer((__gm__ half*)hi); outL.SetGlobalBuffer((__gm__ half*)lo);
    for(int row=GetBlockIdx();row<mp;row+=GetBlockNum()) {
        for(int base=0;base<kp;base+=tile) {
            int allocated=kp-base<tile?kp-base:tile;
            int count=row<o?(kr-base<allocated?kr-base:allocated):0;
            Duplicate(dh,(half)0,tile); Duplicate(dl,(half)0,tile);
            PipeBarrier<PIPE_ALL>();
            if(count>0) {
                int start=row*kr+base,aligned=start/block*block,skew=start-aligned;
                int loaded=(skew+count+block-1)/block*block;
                int split=(loaded+63)/64*64;
                Duplicate(raw,(T)0,split);
                PipeBarrier<PIPE_ALL>();
                DataCopy(raw,input[aligned],loaded); input_ready();
                if constexpr(std::is_same<T,float>::value) {
                    split_float(h,l,raw,tmpBuf.Get<float>(),split);
                } else {
                    Adds(h,raw,(half)0,split); PipeBarrier<PIPE_V>();
                }
                Adds(dh,h[skew],(half)0,count);
                if constexpr(std::is_same<T,float>::value) Adds(dl,l[skew],(half)0,count);
                PipeBarrier<PIPE_V>();
            }
            output_ready();
            DataCopy(outH[row*kp+base],dh,allocated);
            if constexpr(std::is_same<T,float>::value) DataCopy(outL[row*kp+base],dl,allocated);
            output_done();
        }
    }
}

template<class Out,class Bias>
__global__ __aicore__ void finish_kernel(GM_ADDR source,GM_ADDR bias,GM_ADDR target,int o,int n,int np,bool hasBias)
{
    constexpr int tile=2048;
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf,dstBuf,biasBuf,outBuf;
    pipe.InitBuffer(srcBuf,(tile+64)*4); pipe.InitBuffer(dstBuf,tile*4);
    pipe.InitBuffer(biasBuf,32); pipe.InitBuffer(outBuf,tile*sizeof(Out));
    auto src=srcBuf.Get<float>(),dst=dstBuf.Get<float>(); auto b=biasBuf.Get<Bias>();
    GlobalTensor<float> input; input.SetGlobalBuffer((__gm__ float*)source);
    GlobalTensor<Bias> biases; biases.SetGlobalBuffer((__gm__ Bias*)bias);
    GlobalTensor<Out> output; output.SetGlobalBuffer((__gm__ Out*)target);
    for(int base=GetBlockIdx()*tile;base<o*n;base+=GetBlockNum()*tile) {
        int count=o*n-base<tile?o*n-base:tile;
        Duplicate(dst,0.0f,tile); PipeBarrier<PIPE_ALL>();
        for(int offset=0;offset<count;) {
            int pos=base+offset,c=pos/n,col=pos%n;
            int len=n-col<count-offset?n-col:count-offset;
            int start=c*np+col,aligned=start/8*8,skew=start-aligned;
            DataCopy(src,input[aligned],(skew+len+7)/8*8);
            if(hasBias) DataCopy(b,biases[c/(32/sizeof(Bias))*(32/sizeof(Bias))],32/sizeof(Bias));
            input_ready();
            float value=hasBias?(float)b.GetValue(c%(32/sizeof(Bias))):0.0f;
            Adds(dst[offset],src[skew],value,len); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            offset+=len;
        }
        if constexpr(std::is_same<Out,half>::value) {
            Cast(outBuf.Get<half>(),dst,RoundMode::CAST_NONE,64,(count+63)/64,{1,1,4,8});
            PipeBarrier<PIPE_V>(); output_ready();
            DataCopy(output[base],outBuf.Get<half>(),(count+15)/16*16);
        } else {
            output_ready(); DataCopy(output[base],dst,(count+7)/8*8);
        }
        output_done();
    }
}

template<class T,bool nz>
void launch(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& y,
    const std::optional<torch::Tensor>& bias,const torch::Tensor& wh,const torch::Tensor& wl,
    const torch::Tensor& bh,const torch::Tensor& bl,const torch::Tensor& result,
    int stride,int pad,int oh,int ow,int np,int kp,int mp,bool planeFast,bool pointwise,
    bool directWeight,TCubeTiling td,aclrtStream stream)
{
    int c=x.size(1),h=x.size(2),width=x.size(3),o=w.size(0),kernel=w.size(2);
    if(pointwise) {
        if constexpr(std::is_same<T,float>::value)
            split_weights_kernel<<<cores,nullptr,stream>>>((GM_ADDR)x.data_ptr(),(GM_ADDR)bh.data_ptr(),(GM_ADDR)bl.data_ptr(),c*np);
    } else if(planeFast) {
        im2col_kernel<T,nz><<<cores,nullptr,stream>>>((GM_ADDR)x.data_ptr(),(GM_ADDR)bh.data_ptr(),(GM_ADDR)bl.data_ptr(),c,h,width,kp);
    } else {
        tiled_im2col_kernel<T,nz><<<cores,nullptr,stream>>>((GM_ADDR)x.data_ptr(),(GM_ADDR)bh.data_ptr(),(GM_ADDR)bl.data_ptr(),c,h,width,kernel,stride,pad,oh,ow,np,kp);
    }
    if(!directWeight) {
        if constexpr(std::is_same<T,float>::value) {
            if(c*kernel*kernel==kp && o==mp)
                split_weights_kernel<<<cores,nullptr,stream>>>((GM_ADDR)w.data_ptr(),(GM_ADDR)wh.data_ptr(),(GM_ADDR)wl.data_ptr(),o*kp);
            else pack_weights_kernel<T><<<cores,nullptr,stream>>>((GM_ADDR)w.data_ptr(),(GM_ADDR)wh.data_ptr(),(GM_ADDR)wl.data_ptr(),o,c*kernel*kernel,kp,mp);
        } else {
            pack_weights_kernel<T><<<cores,nullptr,stream>>>((GM_ADDR)w.data_ptr(),(GM_ADDR)wh.data_ptr(),(GM_ADDR)wl.data_ptr(),o,c*kernel*kernel,kp,mp);
        }
    }
    conv_mm_kernel<std::is_same<T,float>::value,nz,float><<<cores,nullptr,stream>>>(
        (GM_ADDR)wh.data_ptr(),(GM_ADDR)wl.data_ptr(),(GM_ADDR)bh.data_ptr(),(GM_ADDR)bl.data_ptr(),(GM_ADDR)result.data_ptr(),td,mp,np,kp,mp<64?mp:64);
    GM_ADDR b=bias?(GM_ADDR)bias->data_ptr():nullptr;
    if(y.scalar_type()==torch::kHalf) finish_kernel<half,T><<<cores,nullptr,stream>>>((GM_ADDR)result.data_ptr(),b,(GM_ADDR)y.data_ptr(),o,oh*ow,np,bias.has_value());
    else finish_kernel<float,T><<<cores,nullptr,stream>>>((GM_ADDR)result.data_ptr(),b,(GM_ADDR)y.data_ptr(),o,oh*ow,np,bias.has_value());
}

torch::Tensor conv(const torch::Tensor& x,const torch::Tensor& w,const std::optional<torch::Tensor>& bias,
                   int64_t stride,int64_t pad,bool fp32Output)
{
    if (x.scalar_type() == at::kHalf)
        return image_conv::conv(x, w, bias, stride, pad, fp32Output, false);
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && w.device()==x.device(),"input and weight must be on the same NPU");
    TORCH_CHECK(!x.requires_grad() && !w.requires_grad(),"conv2d is inference-only");
    TORCH_CHECK(x.storage_offset()==0 && w.storage_offset()==0,"conv2d requires allocation-base tensors");
    TORCH_CHECK(x.dim()==4 && w.dim()==4 && x.size(0)==1 && w.size(1)==x.size(1),"expected N=1 NCHW/OIHW convolution");
    int c=x.size(1),h=x.size(2),width=x.size(3),o=w.size(0),kernel=w.size(2);
    TORCH_CHECK((kernel==1 || kernel==3) && w.size(3)==kernel && (stride==1 || stride==2) && pad==kernel/2,"expected 1x1/p0 or 3x3/p1, stride 1 or 2");
    TORCH_CHECK(c>0 && h>0 && width>0 && o>0,"C/H/W/Cout must be positive");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x.scalar_type()==w.scalar_type(),"expected contiguous tensors with matching dtype");
    TORCH_CHECK(x.scalar_type()==torch::kFloat || x.scalar_type()==torch::kHalf,"expected float32 or float16");
    if(bias) TORCH_CHECK(bias->device()==x.device() && bias->scalar_type()==x.scalar_type() && bias->is_contiguous() && bias->storage_offset()==0 && bias->dim()==1 && bias->numel()==o && !bias->requires_grad(),"bias must be contiguous Cout-vector on the same device and dtype");
    thin::select(x.device());
    int oh=(h+2*pad-kernel)/stride+1,ow=(width+2*pad-kernel)/stride+1;
    int n=oh*ow,np=kernel==3 && n>1024?(n+tileN-1)/tileN*tileN:(n+15)/16*16,kr=c*kernel*kernel,kp=(kr+15)/16*16,mp=(o+15)/16*16;
    bool isFloat=x.scalar_type()==torch::kFloat;
    int padded=(n+63)/64*64,halo=(width+31)/16*16;
    int ubBytes=isFloat?16*padded+8*halo:4*padded+4*halo;
    int rowGroup=1; while(width*rowGroup%16!=0) ++rowGroup;
    bool planeFast=kernel==3 && stride==1 && n==np && ubBytes<=262144 && (!isFloat || padded/64<=255) && (padded+2*halo+127)/128<=255 && width*rowGroup/16<=255 && (h+rowGroup-1)/rowGroup<=255;
    bool pointwise=kernel==1 && stride==1 && n==np && c==kp;
    bool nz=!pointwise && np<=1024;
    bool directWeight=!isFloat && kr==kp && o==mp;
    int capacity=(2048*stride*stride+(kernel+stride)*width+191)/64*64;
    int gatherStride=(capacity/2+63)/64*64;
    int gatherCapacity=2*gatherStride>6272?2*gatherStride:6272;
    TORCH_CHECK(planeFast || pointwise || (int(x.element_size())+8)*capacity+8704+2*gatherCapacity<=262144,
                "image width exceeds the tiled im2col UB capacity");
    auto opts=x.options().dtype(torch::kHalf);
    auto bh=pointwise && !isFloat?x:torch::empty({kp,np},opts);
    auto bl=isFloat?torch::empty({kp,np},opts):bh;
    auto wh=directWeight?w:torch::empty({mp,kp},opts);
    auto wl=isFloat?torch::empty({mp,kp},opts):wh;
    auto result=torch::empty({mp,np},x.options().dtype(torch::kFloat));
    auto y=torch::empty({1,o,oh,ow},x.options().dtype(isFloat || fp32Output?torch::kFloat:torch::kHalf));
    auto td=make_tiling(mp,np,kp,nz);
    auto stream=thin::stream();
    if(isFloat) {
        if(nz) launch<float,true>(x,w,y,bias,wh,wl,bh,bl,result,stride,pad,oh,ow,np,kp,mp,planeFast,pointwise,directWeight,td,stream);
        else launch<float,false>(x,w,y,bias,wh,wl,bh,bl,result,stride,pad,oh,ow,np,kp,mp,planeFast,pointwise,directWeight,td,stream);
    } else {
        if(nz) launch<half,true>(x,w,y,bias,wh,wl,bh,bl,result,stride,pad,oh,ow,np,kp,mp,planeFast,pointwise,directWeight,td,stream);
        else launch<half,false>(x,w,y,bias,wh,wl,bh,bl,result,stride,pad,oh,ow,np,kp,mp,planeFast,pointwise,directWeight,td,stream);
    }
    thin::sync();
    return y;
}
}
