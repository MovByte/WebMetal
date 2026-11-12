// Metal's pipeline reflection objects, built from the host's reflection (wire/schema/webmetal.toml, Reflection).
//
// On 21H461 the driver receives a pipeline's reflection from its back-end compiler as serialized AirReflection data and
// hands it to Metal.framework's deserializer (-[MTLComputePipelineReflectionInternal initWithSerializedData:
// serializedStageInputDescriptor:device:options:flags:], Metal 0x187abd5ec; the render reflection's
// initWithVertexData:fragmentData:serializedVertexDescriptor:device:options:flags:), which builds the binding and type
// objects with their classes' initializers. The plugin builds the same objects with the same initializers from the
// host's reflection, then checks every public property of each object against the host's values, so a framework that
// derives a property differently is caught instead of answered.

#import "WebMetalShaders.h"

#import <objc/message.h>
#import <objc/runtime.h>

namespace webmetal {

namespace {

// MTLDataType values (iPhoneOS SDK, MTLArgument.h).
constexpr uint64_t kStruct = 1;
constexpr uint64_t kArray = 2;
constexpr uint64_t kTexture = 58;
constexpr uint64_t kPointer = 60;
constexpr uint32_t kNoType = 0xffffffffu;

// MTLArgumentType values (MTLArgument.h).
constexpr uint64_t kBufferArgument = 0;
constexpr uint64_t kThreadgroupArgument = 1;
constexpr uint64_t kTextureArgument = 2;
constexpr uint64_t kSamplerArgument = 3;

// The value the deserializer gives a struct member's render target and raster order group when the member has none
// (deserializeStructArgument, Metal 0x187a7b034: -1 for an absent field).
constexpr uint64_t kNoAttachment = 0xffffffffu;

// The initializers and setters of Metal.framework's reflection classes, with their 21H461 type encodings (checked by
// CheckReflectionEntryPoints).
using InitBuffer = id (*)(id, SEL, NSString *, uint64_t, uint64_t, BOOL, uint64_t, uint64_t, uint64_t, uint64_t,
                          uint64_t, BOOL, uint64_t, uint64_t);
using InitTexture = id (*)(id, SEL, NSString *, uint64_t, BOOL, uint64_t, uint64_t, uint64_t, uint64_t, BOOL);
using InitThreadgroup = id (*)(id, SEL, NSString *, uint64_t, uint64_t, BOOL, uint64_t, uint64_t, uint64_t, uint64_t,
                               uint64_t);
using InitBinding = id (*)(id, SEL, NSString *, uint64_t, uint64_t, uint64_t, BOOL, uint64_t);
using InitStruct = id (*)(id, SEL, __unsafe_unretained id *, uint64_t);
using InitMember = id (*)(id, SEL, NSString *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, id);
using InitArray = id (*)(id, SEL, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, id);
using InitPointer = id (*)(id, SEL, uint64_t, id, uint64_t, uint64_t, uint64_t, BOOL, BOOL);
using InitTextureType = id (*)(id, SEL, uint64_t, uint64_t, uint64_t, BOOL);
using InitBuiltin = id (*)(id, SEL, NSString *, uint64_t, uint64_t, BOOL, uint64_t, uint64_t, uint64_t);
using SetBool = void (*)(id, SEL, BOOL);
using SetObjectBool = void (*)(id, SEL, id, BOOL);
using SetUInt = void (*)(id, SEL, uint64_t);

struct EntryPoint {
    const char *cls;
    const char *selector;
    const char *types;
};

const EntryPoint kReflectionEntryPoints[] = {
    {"MTLBufferBindingInternal",
     "initWithName:type:access:isActive:locationIndex:arraySize:dataType:pixelFormat:aluType:isConstantBuffer:dataSize:"
     "alignment:",
     "@104@0:8@16Q24Q32B40Q44Q52Q60Q68Q76B84Q88Q96"},
    {"MTLBufferBindingInternal", "setStructType:doRetain:", "v28@0:8@16B24"},
    {"MTLBufferBindingInternal", "setVertexDescriptorBuffer:", "v20@0:8B16"},
    {"MTLBindingInternal", "setIsArgument:", "v20@0:8B16"},
    {"MTLBindingInternal", "initWithName:type:access:index:active:arrayLength:", "@60@0:8@16Q24Q32Q40B48Q52"},
    {"MTLTextureBindingInternal", "initWithName:access:isActive:locationIndex:arraySize:dataType:textureType:isDepthTexture:",
     "@72@0:8@16Q24B32Q36Q44Q52Q60B68"},
    {"MTLThreadgroupMemoryBindingInternal",
     "initWithName:type:access:isActive:locationIndex:arraySize:dataType:dataSize:alignment:",
     "@84@0:8@16Q24Q32B40Q44Q52Q60Q68Q76"},
    {"MTLStructTypeInternal", "initWithMembers:count:", "@32@0:8^@16Q24"},
    {"MTLStructTypeInternal", "setIsIndirectArgumentBuffer:", "v20@0:8B16"},
    {"MTLStructMemberInternal",
     "initWithName:offset:dataType:pixelFormat:aluType:indirectArgumentIndex:render_target:raster_order_group:details:",
     "@88@0:8@16Q24Q32Q40Q48Q56Q64Q72@80"},
    {"MTLArrayTypeInternal", "initWithArrayLength:elementType:stride:pixelFormat:aluType:details:",
     "@64@0:8Q16Q24Q32Q40Q48@56"},
    {"MTLArrayTypeInternal", "setArgumentIndexStride:", "v24@0:8Q16"},
    {"MTLPointerTypeInternal",
     "initWithElementType:elementTypeDescription:access:alignment:dataSize:elementIsIndirectArgumentBuffer:"
     "isConstantBuffer:",
     "@64@0:8Q16@24Q32Q40Q48B56B60"},
    {"MTLTextureReferenceTypeInternal", "initWithDataType:textureType:access:isDepthTexture:", "@44@0:8Q16Q24Q32B40"},
    {"MTLBuiltInArgument", "initWithName:type:access:active:index:dataType:builtInType:", "@68@0:8@16Q24Q32B40Q44Q52Q60"},
};

void requireReflectionEntryPoints(id self, SEL _cmd)
{
    static std::string missing;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        for (const EntryPoint &e : kReflectionEntryPoints) {
            Method method = class_getInstanceMethod(objc_getClass(e.cls), sel_registerName(e.selector));
            const char *types = method ? method_getTypeEncoding(method) : nullptr;
            if (!types || strcmp(types, e.types) != 0) {
                missing = std::string("-[") + e.cls + " " + e.selector + "] " +
                          (types ? std::string("has type encoding ") + types : "is missing") + ", expected " + e.types;
                return;
            }
        }
    });
    if (!missing.empty())
        WMLoudMethod(self, _cmd, ("refused: the running Metal.framework lacks a reflection initializer: " + missing).c_str());
}

template <typename F> F imp(const char *cls, const char *selector)
{
    return (F)class_getMethodImplementation(objc_getClass(cls), sel_registerName(selector));
}

NSString *string(const std::string &s)
{
    return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];
}

// Builds the MTLType objects of one stage, each once.
class Types {
public:
    Types(id self, SEL cmd, const wire::StageReflection &stage) : self_(self), cmd_(cmd), stage_(stage),
        built_(stage.types.size()) {}

    id get(uint32_t index)
    {
        if (index == kNoType)
            return nil;
        if (index >= stage_.types.size())
            fail("a type index outside the stage's types");
        if (!built_[index])
            built_[index] = build(stage_.types[index]);
        return built_[index];
    }

    const wire::TypeReflection &at(uint32_t index)
    {
        if (index >= stage_.types.size())
            fail("a type index outside the stage's types");
        return stage_.types[index];
    }

    // Whether an AIR address space is the constant one (1 device, 2 constant); a host that does not report it leaves
    // Metal.framework's isConstantBuffer unknown, and the object is not built.
    bool constantSpace(uint32_t space) const
    {
        if (space == 1 || space == 2)
            return space == 2;
        WMLoudMethod(self_, cmd_, space ? ("the host reports address space " + std::to_string(space) + " for a buffer")
                                                  .c_str()
                                        : "the host does not report a buffer's address space, which Metal.framework's "
                                          "MTLPointerType records (isConstantBuffer)");
    }

    [[noreturn]] void fail(const std::string &what) const
    {
        WMLoudMethod(self_, cmd_, ("the host's reflection does not describe a Metal type: " + what).c_str());
    }

    // A built object's public property differs from the host's value.
    void check(bool same, const char *what) const
    {
        if (!same)
            WMLoudMethod(self_, cmd_, (std::string("Metal.framework derives the reflection's ") + what +
                                       " differently from the host's value").c_str());
    }

private:
    id build(const wire::TypeReflection &t)
    {
        switch (t.dataType) {
        case kStruct: {
            NSMutableArray *members = [NSMutableArray array];
            for (const wire::StructMemberReflection &m : t.members) {
                id details = get(m.type);
                id member = imp<InitMember>("MTLStructMemberInternal",
                                            "initWithName:offset:dataType:pixelFormat:aluType:indirectArgumentIndex:"
                                            "render_target:raster_order_group:details:")(
                    [objc_getClass("MTLStructMemberInternal") alloc],
                    sel_registerName("initWithName:offset:dataType:pixelFormat:aluType:indirectArgumentIndex:"
                                     "render_target:raster_order_group:details:"),
                    string(m.name), m.offset, m.dataType, 0, 0, m.argumentIndex, kNoAttachment, kNoAttachment, details);
                MTLStructMember *public_ = member;
                check([public_.name isEqualToString:string(m.name)] && public_.offset == m.offset &&
                          public_.dataType == m.dataType && public_.argumentIndex == m.argumentIndex,
                      "struct member");
                [members addObject:member];
            }
            std::vector<__unsafe_unretained id> list(members.count);
            for (NSUInteger i = 0; i < members.count; i++)
                list[i] = members[i];
            return imp<InitStruct>("MTLStructTypeInternal", "initWithMembers:count:")(
                [objc_getClass("MTLStructTypeInternal") alloc], sel_registerName("initWithMembers:count:"), list.data(),
                list.size());
        }
        case kArray: {
            id element = get(t.elementType);
            id array = imp<InitArray>("MTLArrayTypeInternal",
                                      "initWithArrayLength:elementType:stride:pixelFormat:aluType:details:")(
                [objc_getClass("MTLArrayTypeInternal") alloc],
                sel_registerName("initWithArrayLength:elementType:stride:pixelFormat:aluType:details:"), t.arrayLength,
                t.elementDataType, t.stride, 0, 0, element);
            imp<SetUInt>("MTLArrayTypeInternal", "setArgumentIndexStride:")(
                array, sel_registerName("setArgumentIndexStride:"), t.argumentIndexStride);
            MTLArrayType *public_ = array;
            check(public_.arrayLength == t.arrayLength && public_.elementType == t.elementDataType &&
                      public_.stride == t.stride && public_.argumentIndexStride == t.argumentIndexStride,
                  "array type");
            return array;
        }
        case kPointer: {
            const bool constant = constantSpace(t.addressSpace);
            id element = get(t.elementType);
            if (element && t.elementIsArgumentBuffer)
                markArgumentBuffer(element);
            id pointer = imp<InitPointer>("MTLPointerTypeInternal",
                                          "initWithElementType:elementTypeDescription:access:alignment:dataSize:"
                                          "elementIsIndirectArgumentBuffer:isConstantBuffer:")(
                [objc_getClass("MTLPointerTypeInternal") alloc],
                sel_registerName("initWithElementType:elementTypeDescription:access:alignment:dataSize:"
                                 "elementIsIndirectArgumentBuffer:isConstantBuffer:"),
                t.elementDataType, element, t.access, t.alignment, t.dataSize, t.elementIsArgumentBuffer, constant);
            checkPointer(pointer, t);
            return pointer;
        }
        case kTexture: {
            id texture = imp<InitTextureType>("MTLTextureReferenceTypeInternal",
                                              "initWithDataType:textureType:access:isDepthTexture:")(
                [objc_getClass("MTLTextureReferenceTypeInternal") alloc],
                sel_registerName("initWithDataType:textureType:access:isDepthTexture:"), t.textureDataType,
                t.textureType, t.access, t.isDepthTexture);
            MTLTextureReferenceType *public_ = texture;
            check(public_.textureDataType == t.textureDataType && public_.textureType == t.textureType &&
                      public_.access == t.access && public_.isDepthTexture == t.isDepthTexture,
                  "texture reference type");
            return texture;
        }
        default:
            fail("data type " + std::to_string(t.dataType) + " as a type of its own");
        }
    }

    // The struct an argument-buffer pointer points to is an argument buffer (-[MTLStructTypeInternal
    // setIsIndirectArgumentBuffer:]); the pointer's elementIsArgumentBuffer is derived from it.
    void markArgumentBuffer(id element)
    {
        if (![element isKindOfClass:objc_getClass("MTLStructTypeInternal")])
            fail("an argument-buffer pointer to something other than a struct");
        imp<SetBool>("MTLStructTypeInternal", "setIsIndirectArgumentBuffer:")(
            element, sel_registerName("setIsIndirectArgumentBuffer:"), YES);
    }

public:
    void checkPointer(id pointer, const wire::TypeReflection &t) const
    {
        MTLPointerType *public_ = pointer;
        check(public_.elementType == t.elementDataType && public_.access == t.access &&
                  public_.alignment == t.alignment && public_.dataSize == t.dataSize &&
                  public_.elementIsArgumentBuffer == t.elementIsArgumentBuffer,
              "pointer type");
    }

private:
    id self_;
    SEL cmd_;
    const wire::StageReflection &stage_;
    std::vector<id> built_;
};

id newBinding(id self, SEL _cmd, Types &types, const wire::ArgumentReflection &a, bool vertexDescriptorBuffer)
{
    NSString *name = string(a.name);
    id binding = nil;
    switch (a.type) {
    case kBufferArgument: {
        binding = imp<InitBuffer>("MTLBufferBindingInternal",
                                  "initWithName:type:access:isActive:locationIndex:arraySize:dataType:pixelFormat:"
                                  "aluType:isConstantBuffer:dataSize:alignment:")(
            [objc_getClass("MTLBufferBindingInternal") alloc],
            sel_registerName("initWithName:type:access:isActive:locationIndex:arraySize:dataType:pixelFormat:aluType:"
                             "isConstantBuffer:dataSize:alignment:"),
            name, a.type, a.access, a.active, a.index, a.arrayLength, a.bufferDataType, 0, 0,
            types.constantSpace(a.bufferAddressSpace), a.bufferDataSize, a.bufferAlignment);
        if (a.bufferStructType != kNoType) {
            id structType = types.get(a.bufferStructType);
            if (a.bufferPointerType != kNoType && types.at(a.bufferPointerType).elementIsArgumentBuffer)
                imp<SetBool>("MTLStructTypeInternal", "setIsIndirectArgumentBuffer:")(
                    structType, sel_registerName("setIsIndirectArgumentBuffer:"), YES);
            imp<SetObjectBool>("MTLBufferBindingInternal", "setStructType:doRetain:")(
                binding, sel_registerName("setStructType:doRetain:"), structType, YES);
        }
        if (vertexDescriptorBuffer)
            imp<SetBool>("MTLBufferBindingInternal", "setVertexDescriptorBuffer:")(
                binding, sel_registerName("setVertexDescriptorBuffer:"), YES);
        id<MTLBufferBinding> buffer = binding;
        types.check(buffer.bufferAlignment == a.bufferAlignment && buffer.bufferDataSize == a.bufferDataSize &&
                        buffer.bufferDataType == a.bufferDataType,
                    "buffer binding");
        if (a.bufferPointerType == kNoType)
            types.fail("a buffer argument without a pointer type");
        const wire::TypeReflection &pointer = types.at(a.bufferPointerType);
        if (pointer.addressSpace != a.bufferAddressSpace)
            types.fail("a buffer whose pointer type and argument disagree on the address space");
        types.checkPointer(buffer.bufferPointerType, pointer);
        types.check((buffer.bufferStructType != nil) == (a.bufferStructType != kNoType), "buffer struct type");
        break;
    }
    case kThreadgroupArgument: {
        if (a.threadgroupMemoryDataType == 0)
            WMLoudMethod(self, _cmd, "the host does not report the data type of threadgroup memory, which Metal.framework's "
                                     "MTLThreadgroupMemoryBindingInternal records");
        if (a.threadgroupMemoryDataType == kStruct)
            types.fail("threadgroup memory of a struct type, whose struct the wire does not carry");
        binding = imp<InitThreadgroup>("MTLThreadgroupMemoryBindingInternal",
                                       "initWithName:type:access:isActive:locationIndex:arraySize:dataType:dataSize:"
                                       "alignment:")(
            [objc_getClass("MTLThreadgroupMemoryBindingInternal") alloc],
            sel_registerName("initWithName:type:access:isActive:locationIndex:arraySize:dataType:dataSize:alignment:"),
            name, a.type, a.access, a.active, a.index, a.arrayLength, a.threadgroupMemoryDataType,
            a.threadgroupMemoryDataSize, a.threadgroupMemoryAlignment);
        id<MTLThreadgroupBinding> threadgroup = binding;
        types.check(threadgroup.threadgroupMemoryAlignment == a.threadgroupMemoryAlignment &&
                        threadgroup.threadgroupMemoryDataSize == a.threadgroupMemoryDataSize,
                    "threadgroup binding");
        break;
    }
    case kTextureArgument: {
        binding = imp<InitTexture>("MTLTextureBindingInternal",
                                   "initWithName:access:isActive:locationIndex:arraySize:dataType:textureType:"
                                   "isDepthTexture:")(
            [objc_getClass("MTLTextureBindingInternal") alloc],
            sel_registerName("initWithName:access:isActive:locationIndex:arraySize:dataType:textureType:isDepthTexture:"),
            name, a.access, a.active, a.index, a.arrayLength, a.textureDataType, a.textureType, a.isDepthTexture);
        id<MTLTextureBinding> texture = binding;
        types.check(texture.textureDataType == a.textureDataType && texture.textureType == a.textureType &&
                        texture.depthTexture == a.isDepthTexture && texture.arrayLength == a.arrayLength,
                    "texture binding");
        break;
    }
    case kSamplerArgument:
        binding = imp<InitBinding>("MTLBindingInternal", "initWithName:type:access:index:active:arrayLength:")(
            [objc_getClass("MTLBindingInternal") alloc],
            sel_registerName("initWithName:type:access:index:active:arrayLength:"), name, a.type, a.access, a.index,
            a.active, a.arrayLength);
        break;
    default:
        WMLoudMethod(self, _cmd, ("a reflected argument of MTLArgumentType " + std::to_string(a.type) +
                                  ", which the A10X's GPU family does not have and the plugin does not build")
                                     .c_str());
    }
    imp<SetBool>("MTLBindingInternal", "setIsArgument:")(binding, sel_registerName("setIsArgument:"),
                                                          !vertexDescriptorBuffer);
    id<MTLBinding> common = binding;
    types.check([common.name isEqualToString:name] && (uint64_t)common.type == a.type && common.access == a.access &&
                    common.index == a.index && common.used == a.active && common.argument == !vertexDescriptorBuffer,
                "binding");
    return binding;
}

NSArray *builtins(id self, SEL _cmd, const wire::StageReflection &stage)
{
    if (!stage.builtins)
        WMLoudMethod(self, _cmd, "the host does not report the stage's built-in arguments, which Metal.framework's "
                                 "reflection lists (builtInArguments)");
    NSMutableArray *out = [NSMutableArray array];
    for (const wire::BuiltinReflection &b : stage.builtins->list) {
        id builtin = imp<InitBuiltin>("MTLBuiltInArgument", "initWithName:type:access:active:index:dataType:builtInType:")(
            [objc_getClass("MTLBuiltInArgument") alloc],
            sel_registerName("initWithName:type:access:active:index:dataType:builtInType:"), string(b.name), b.type,
            b.access, b.active, b.index, b.dataType, b.builtinType);
        [out addObject:builtin];
    }
    return out;
}

// The bindings of one stage, in the host's order. The first `vertexBuffers` arguments are the buffers the vertex
// descriptor feeds (named vertexBuffer.<n>, which no MSL argument can be named), which Metal reports as bindings that
// are not function arguments.
NSArray *bindings(id self, SEL _cmd, const wire::StageReflection &stage, size_t vertexBuffers)
{
    Types types(self, _cmd, stage);
    NSMutableArray *out = [NSMutableArray array];
    for (size_t i = 0; i < stage.arguments.size(); i++) {
        const bool synthesized = i < vertexBuffers;
        if (synthesized && stage.arguments[i].name.rfind("vertexBuffer.", 0) != 0)
            types.fail("the vertex descriptor's buffer " + std::to_string(i) + " is reported as " +
                       stage.arguments[i].name);
        [out addObject:newBinding(self, _cmd, types, stage.arguments[i], synthesized)];
    }
    return out;
}

void setIvar(id object, const char *name, id value)
{
    Ivar ivar = class_getInstanceVariable(object_getClass(object), name);
    object_setIvarWithStrongDefault(object, ivar, value);
}

// The buffers a vertex descriptor's attributes read: one per distinct buffer index of an attribute with a format.
size_t vertexDescriptorBuffers(MTLVertexDescriptor *descriptor)
{
    if (!descriptor)
        return 0;
    NSMutableIndexSet *buffers = [NSMutableIndexSet indexSet];
    for (NSUInteger i = 0; i < 31; i++) {
        MTLVertexAttributeDescriptor *a = descriptor.attributes[i];
        if (a.format != MTLVertexFormatInvalid)
            [buffers addIndex:a.bufferIndex];
    }
    return buffers.count;
}

}  // namespace

MTLComputePipelineReflection *NewComputeReflection(id self, SEL _cmd, const wire::Reflection &reflection)
{
    requireReflectionEntryPoints(self, _cmd);
    if (reflection.stages.size() != 1 || reflection.stages[0].functionType != MTLFunctionTypeKernel)
        WMLoudMethod(self, _cmd, "the host's reflection of a compute pipeline is not one kernel stage");
    const wire::StageReflection &stage = reflection.stages[0];
    id out = [objc_getClass("MTLComputePipelineReflectionInternal") new];
    NSArray *list = bindings(self, _cmd, stage, 0);
    setIvar(out, "_arguments", [list mutableCopy]);
    setIvar(out, "_bindings", [list copy]);
    setIvar(out, "_builtInArguments", builtins(self, _cmd, stage));
    return out;
}

MTLRenderPipelineReflection *NewRenderReflection(id self, SEL _cmd, const wire::Reflection &reflection,
                                                 MTLVertexDescriptor *vertexDescriptor)
{
    requireReflectionEntryPoints(self, _cmd);
    id out = [objc_getClass("MTLRenderPipelineReflectionInternal") new];
    bool vertex = false;
    for (const wire::StageReflection &stage : reflection.stages) {
        if (stage.functionType == MTLFunctionTypeVertex && !vertex) {
            vertex = true;
            NSArray *list = bindings(self, _cmd, stage, vertexDescriptorBuffers(vertexDescriptor));
            setIvar(out, "_vertexArguments", [list mutableCopy]);
            setIvar(out, "_vertexBindings", [list copy]);
            setIvar(out, "_vertexBuiltInArguments", builtins(self, _cmd, stage));
        } else if (stage.functionType == MTLFunctionTypeFragment) {
            NSArray *list = bindings(self, _cmd, stage, 0);
            setIvar(out, "_fragmentArguments", [list mutableCopy]);
            setIvar(out, "_fragmentBindings", [list copy]);
            // The 21H461 class keeps no fragment built-in arguments (it has vertex, tile, object and mesh ones).
        } else {
            WMLoudMethod(self, _cmd, "the host's reflection of a render pipeline has a stage other than one vertex "
                                     "and one fragment stage");
        }
    }
    if (!vertex)
        WMLoudMethod(self, _cmd, "the host's reflection of a render pipeline has no vertex stage");
    return out;
}

}  // namespace webmetal
