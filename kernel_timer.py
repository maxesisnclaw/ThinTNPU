import ctypes as C
import os
import statistics
import torch


def time_nms_kernel(boxes, scores, iou_threshold, score_threshold, max_out, repeats=7):
    lib=C.CDLL('/usr/local/Ascend/ascend-toolkit/latest/lib64/libacl_rt.so')
    ptr=C.c_void_p
    signatures={
        'aclrtCreateStream':[C.POINTER(ptr)], 'aclrtDestroyStream':[ptr],
        'aclrtBinaryLoadFromFile':[C.c_char_p,ptr,C.POINTER(ptr)],
        'aclrtBinaryGetFunction':[ptr,C.c_char_p,C.POINTER(ptr)],
        'aclrtCreateEvent':[C.POINTER(ptr)], 'aclrtDestroyEvent':[ptr],
        'aclrtRecordEvent':[ptr,ptr], 'aclrtEventElapsedTime':[C.POINTER(C.c_float),ptr,ptr],
        'aclrtLaunchKernelWithHostArgs':[ptr,C.c_uint32,ptr,ptr,ptr,C.c_size_t,ptr,C.c_uint32],
        'aclrtSynchronizeStream':[ptr], 'aclrtBinaryUnLoad':[ptr],
    }
    for name,args in signatures.items():
        fn=getattr(lib,name); fn.argtypes=args; fn.restype=C.c_int
    def call(name,*args):
        result=getattr(lib,name)(*args)
        if result: raise RuntimeError(name+' returned '+str(result))
    stream=ptr(); binary=ptr(); fn=ptr()
    call('aclrtCreateStream',C.byref(stream))
    call('aclrtBinaryLoadFromFile',os.environ['THIN_NMS_BINARY'].encode(),None,C.byref(binary))
    call('aclrtBinaryGetFunction',binary,b'nms_fp32',C.byref(fn))
    length=boxes.shape[0]
    scratch=torch.empty(((length+1023)//1024*6*1024,),device=boxes.device)
    indices=torch.empty((max_out,),dtype=torch.int32,device=boxes.device)
    count=torch.empty((1,),dtype=torch.int32,device=boxes.device)
    class Args(C.Structure):
        _fields_=[('boxes',ptr),('scores',ptr),('scratch',ptr),('indices',ptr),('count',ptr),
                  ('length',C.c_uint64),('max_out',C.c_uint64),('iou',C.c_float),('score_thr',C.c_float)]
    args=Args(boxes.data_ptr(),scores.data_ptr(),scratch.data_ptr(),indices.data_ptr(),count.data_ptr(),
              length,max_out,iou_threshold,score_threshold)
    samples=[]
    for _ in range(repeats):
        start=ptr(); stop=ptr()
        call('aclrtCreateEvent',C.byref(start)); call('aclrtCreateEvent',C.byref(stop))
        call('aclrtRecordEvent',start,stream)
        call('aclrtLaunchKernelWithHostArgs',fn,1,stream,None,C.byref(args),C.sizeof(args),None,0)
        call('aclrtRecordEvent',stop,stream)
        call('aclrtSynchronizeStream',stream)
        ms=C.c_float(); call('aclrtEventElapsedTime',C.byref(ms),start,stop)
        samples.append(ms.value)
        call('aclrtDestroyEvent',start); call('aclrtDestroyEvent',stop)
    n=count.cpu().item()
    kept=indices.cpu().tolist()[:n]
    call('aclrtBinaryUnLoad',binary)
    call('aclrtDestroyStream',stream)
    return dict(kernel_event_ms_median=statistics.median(samples),kernel_event_samples_ms=samples),kept
