#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <vector>
#include <utility>
namespace py = pybind11;

struct TensorRecord {
    py::str name;
    py::object tensor;
    uint32_t version;
    const void* data;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t offset;
    c10::ScalarType dtype;
    c10::Device device;
};
struct CollectionRecord {
    py::str key;
    std::vector<TensorRecord> tensors;
    Py_ssize_t count;
};
struct ModuleRecord {
    py::object module;
    py::object kind;
    py::dict data;
    py::object function;
    py::object owner;
    bool ignoreForward;
    bool defaultForward;
    std::vector<std::pair<py::str, py::object>> children;
    std::vector<std::pair<py::str, py::object>> attributes;
    std::vector<CollectionRecord> collections;
};

class Guard {
    std::vector<ModuleRecord> records;
    py::object registry;
    std::vector<std::pair<py::object,py::object>> classes;
    static py::str key(const char* name) {
        auto value = PyUnicode_InternFromString(name);
        if (!value) throw py::error_already_set();
        return py::reinterpret_steal<py::str>(value);
    }
    const py::str training = key("training"), modulesKey = key("_modules"), forwardKey = key("forward");
    const std::vector<py::str> hooks{key("_forward_hooks"),key("_forward_pre_hooks"),
                                   key("_backward_hooks"),key("_backward_pre_hooks")};
    const std::vector<py::str> globalHooks{key("_global_forward_hooks"),key("_global_forward_pre_hooks"),
                                         key("_global_backward_hooks"),key("_global_backward_pre_hooks")};
    static PyObject* dict(PyObject* value) {
        if (!value || !PyDict_Check(value)) throw py::value_error("Expected a module state dictionary");
        return value;
    }
    static PyObject* get(PyObject* data, PyObject* name) {
        auto value = PyDict_GetItemWithError(data,name);
        if (!value && PyErr_Occurred()) throw py::error_already_set();
        return value;
    }
    static PyObject* required(PyObject* data, PyObject* name) {
        auto value = get(data,name);
        if (!value) throw py::value_error("Region attribute removed");
        return value;
    }
    static Py_ssize_t length(PyObject* value) {
        auto count = PyObject_Length(value);
        if (count < 0) throw py::error_already_set();
        return count;
    }
    static std::pair<py::object,py::object> forward(py::object module) {
        auto method = module.attr("forward");
        if (!PyMethod_Check(method.ptr())) throw py::value_error("Expected a bound module forward");
        return {py::reinterpret_borrow<py::object>(PyMethod_Function(method.ptr())),
                py::reinterpret_borrow<py::object>(PyMethod_Self(method.ptr()))};
    }
public:
    Guard(py::iterable modules, py::set ignored, py::iterable attributes)
        : registry(py::module_::import("torch.nn.modules.module")) {
        for (auto item : modules) {
            auto module = py::reinterpret_borrow<py::object>(item);
            auto data = module.attr("__dict__").cast<py::dict>();
            auto fn = forward(module);
            ModuleRecord record{module,py::type::of(module),data,fn.first,fn.second,
                                ignored.contains(module),get(data.ptr(),forwardKey.ptr()) == nullptr,{},{},{}};
            for (auto pair : py::reinterpret_borrow<py::dict>(dict(required(data.ptr(),modulesKey.ptr()))))
                record.children.emplace_back(py::reinterpret_borrow<py::str>(pair.first),
                                             py::reinterpret_borrow<py::object>(pair.second));
            for (auto name : attributes) {
                auto value = get(data.ptr(),name.ptr());
                if (value) record.attributes.emplace_back(py::reinterpret_borrow<py::str>(name),
                                                          py::reinterpret_borrow<py::object>(value));
            }
            for (const char* collection : {"_parameters","_buffers"}) {
                CollectionRecord values{key(collection),{},0};
                auto current = dict(required(data.ptr(),values.key.ptr()));
                values.count = length(current);
                for (auto pair : py::reinterpret_borrow<py::dict>(current)) {
                    auto tensor = py::reinterpret_borrow<py::object>(pair.second);
                    if (!tensor.is_none() && !THPVariable_Check(tensor.ptr()))
                        throw py::type_error("Expected a Tensor or None in module parameters/buffers");
                    const auto value = tensor.is_none() ? at::Tensor() : THPVariable_Unpack(tensor.ptr());
                    values.tensors.push_back({py::reinterpret_borrow<py::str>(pair.first),tensor,
                        value.defined() ? value.unsafeGetTensorImpl()->version_counter().current_version() : 0,
                        value.defined() ? value.const_data_ptr() : nullptr,
                        value.defined() ? value.sizes().vec() : std::vector<int64_t>{},
                        value.defined() ? value.strides().vec() : std::vector<int64_t>{},
                        value.defined() ? value.storage_offset() : 0,
                        value.defined() ? value.scalar_type() : c10::ScalarType::Undefined,
                        value.defined() ? value.device() : c10::Device(c10::DeviceType::CPU)});
                }
                record.collections.push_back(std::move(values));
            }
            bool known = false;
            for (const auto& item : classes) if (item.first.ptr() == record.kind.ptr()) known = true;
            if (!known) classes.emplace_back(record.kind,record.kind.attr("forward"));
            records.push_back(std::move(record));
        }
        validate();
    }
    void validate() const {
        for (const auto& name : globalHooks) {
            auto value = PyObject_GetAttr(registry.ptr(),name.ptr());
            if (!value) throw py::error_already_set();
            auto count = PyObject_Length(value);
            Py_DECREF(value);
            if (count < 0) throw py::error_already_set();
            if (count) throw py::value_error("Global module hooks are unsupported");
        }
        for (const auto& item : classes) {
            auto value = PyObject_GetAttr(item.first.ptr(),forwardKey.ptr());
            if (!value) throw py::error_already_set();
            bool changed = value != item.second.ptr();
            Py_DECREF(value);
            if (changed) throw py::value_error("Module class forward changed");
        }
        for (const auto& record : records) {
            if (reinterpret_cast<PyObject*>(Py_TYPE(record.module.ptr())) != record.kind.ptr())
                throw py::value_error("Region module class changed");
            auto current = PyObject_GenericGetDict(record.module.ptr(),nullptr);
            if (!current) throw py::error_already_set();
            bool changed = current != record.data.ptr();
            Py_DECREF(current);
            if (changed) throw py::value_error("Region module dictionary changed");
            auto data = record.data.ptr();
            if (required(data,training.ptr()) != Py_False) throw py::value_error("Region is inference-only");
            for (const auto& name : hooks)
                if (length(required(data,name.ptr()))) throw py::value_error("Region hooks changed");
            auto children = dict(required(data,modulesKey.ptr()));
            if (PyDict_Size(children) != static_cast<Py_ssize_t>(record.children.size()))
                throw py::value_error("Region children changed");
            for (const auto& pair : record.children)
                if (get(children,pair.first.ptr()) != pair.second.ptr())
                    throw py::value_error("Region child identity changed");
            if (!record.ignoreForward) {
                auto method = get(data,forwardKey.ptr());
                if (method) {
                    if (!PyMethod_Check(method) || PyMethod_Function(method) != record.function.ptr() ||
                            PyMethod_Self(method) != record.owner.ptr()) throw py::value_error("Region forward changed");
                } else if (!record.defaultForward) throw py::value_error("Region forward changed");
            }
            for (const auto& pair : record.attributes) {
                auto value = get(data,pair.first.ptr());
                if (!value || (value != pair.second.ptr() && PyObject_RichCompareBool(value,pair.second.ptr(),Py_EQ) != 1))
                    throw py::value_error("Region operator configuration changed");
            }
            for (const auto& collection : record.collections) {
                auto valuesOwner = py::reinterpret_borrow<py::object>(dict(required(data,collection.key.ptr())));
                auto values = valuesOwner.ptr();
                if (length(values) != collection.count)
                    throw py::value_error("Region parameter/buffer set changed");
                for (const auto& tensor : collection.tensors) {
                    if (get(values,tensor.name.ptr()) != tensor.tensor.ptr())
                        throw py::value_error("Region parameter/prepack identity changed");
                    if (!tensor.tensor.is_none()) {
                        const auto& value = THPVariable_Unpack(tensor.tensor.ptr());
                        if (value.unsafeGetTensorImpl()->version_counter().current_version() != tensor.version || value.requires_grad() ||
                                value.const_data_ptr() != tensor.data || value.scalar_type() != tensor.dtype || value.device() != tensor.device ||
                                value.storage_offset() != tensor.offset || value.sizes() != at::IntArrayRef(tensor.sizes) ||
                                value.strides() != at::IntArrayRef(tensor.strides))
                            throw py::value_error("Region parameter/prepack version or gradient mode changed");
                    }
                }
            }
        }
    }
};
PYBIND11_MODULE(_region_guard,m) {
    py::class_<Guard>(m,"Guard").def(py::init<py::iterable,py::set,py::iterable>()).def("validate",&Guard::validate);
}
