// The wire forms of pipeline descriptors (wire/schema/webmetal.toml: RenderPipelineDescriptor,
// ComputePipelineDescriptor and the structures they hold). Every field is the settable state of the 21H461 descriptor
// class, read with the class's own getter (the getter names are the 21H461 classes' method names); a collection carries
// one element per slot of the 21H461 array class (31 buffers, attributes and layouts, 8 color attachments).

#import "WebMetalPipelineDescriptors.h"

#import <objc/message.h>

namespace webmetal {

namespace {

// A getter of the running framework's descriptor class. The plugin reads every field the wire carries; a framework that
// lacks one cannot be served.
template <typename T> T get(id self, SEL _cmd, id object, const char *name)
{
    SEL selector = sel_registerName(name);
    if (![object respondsToSelector:selector])
        WMLoudMethod(self, _cmd, (std::string("the descriptor has no -") + name + ", which the wire format carries").c_str());
    return ((T(*)(id, SEL))objc_msgSend)(object, selector);
}

// The slot counts of the 21H461 descriptor arrays (MTLPipelineBufferDescriptorArrayInternal, the vertex and stage-input
// attribute and layout arrays: 31 descriptors each; MTLRenderPipelineColorAttachmentDescriptorArrayInternal: 8).
constexpr NSUInteger kBufferSlots = 31;
constexpr NSUInteger kAttributeSlots = 31;
constexpr NSUInteger kColorSlots = 8;

// A property-list value of a descriptor field: nil, or the dictionary the wire carries.
const wire::Value *value(Storage &s, id self, SEL _cmd, id object, const char *field)
{
    if (!object)
        return nullptr;
    s.values.emplace_back();
    if (!wire::valueFromObject(object, s.values.back()))
        WMLoudMethod(self, _cmd, (std::string("the descriptor's ") + field + " has no property-list form").c_str());
    return &s.values.back();
}

wire::Span<uint64_t> handles(Storage &s, std::vector<uint64_t> list)
{
    s.handles.push_back(std::move(list));
    return {s.handles.back().data(), s.handles.back().size()};
}

wire::Span<uint64_t> functions(Storage &s, WebMetalDevice *device, id self, SEL _cmd, NSArray *list)
{
    std::vector<uint64_t> out;
    for (id<MTLFunction> f in list)
        out.push_back(FunctionHandle(device, self, _cmd, f));
    return handles(s, std::move(out));
}

wire::Span<wire::PipelineBufferDescriptor> buffers(Storage &s, id self, SEL _cmd, id array)
{
    s.buffers.emplace_back();
    std::vector<wire::PipelineBufferDescriptor> &out = s.buffers.back();
    for (NSUInteger i = 0; i < kBufferSlots; i++) {
        MTLPipelineBufferDescriptor *b = [(MTLPipelineBufferDescriptorArray *)array objectAtIndexedSubscript:i];
        wire::PipelineBufferDescriptor w;
        w.mutability = get<NSUInteger>(self, _cmd, b, "mutability");
        out.push_back(w);
    }
    return {out.data(), out.size()};
}

wire::Opt<wire::LinkedFunctions> linked(Storage &s, WebMetalDevice *device, id self, SEL _cmd, MTLLinkedFunctions *in)
{
    wire::Opt<wire::LinkedFunctions> out;
    if (!in)
        return out;
    out.present = true;
    out.value.binaryFunctions = functions(s, device, self, _cmd, get<NSArray *>(self, _cmd, in, "binaryFunctions"));
    out.value.functions = functions(s, device, self, _cmd, get<NSArray *>(self, _cmd, in, "functions"));
    out.value.privateFunctions = functions(s, device, self, _cmd, get<NSArray *>(self, _cmd, in, "privateFunctions"));
    NSDictionary *groups = get<NSDictionary *>(self, _cmd, in, "groups");
    s.groups.emplace_back();
    std::vector<wire::LinkedFunctionGroup> &list = s.groups.back();
    // NSDictionary has no order; the entries travel sorted by group name, as the wire's dictionaries do.
    for (NSString *name in [groups.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
        if (![name isKindOfClass:[NSString class]])
            WMLoudMethod(self, _cmd, "a linked-function group's key is not a string");
        s.strings.push_back(name.UTF8String ?: "");
        wire::LinkedFunctionGroup g;
        g.name = wire::str(s.strings.back());
        g.functions = functions(s, device, self, _cmd, groups[name]);
        list.push_back(g);
    }
    out.value.groups = {list.data(), list.size()};
    return out;
}

wire::Opt<wire::ProfileControl> profile(id self, SEL _cmd, id in)
{
    wire::Opt<wire::ProfileControl> out;
    if (!in)
        return out;
    out.present = true;
    out.value.uscCliqueTraceLevel = get<uint8_t>(self, _cmd, in, "uscCliqueTraceLevel");
    out.value.uscEmitPosition = get<BOOL>(self, _cmd, in, "uscEmitPosition");
    out.value.uscEmitThreadControlFlow = get<BOOL>(self, _cmd, in, "uscEmitThreadControlFlow");
    return out;
}

wire::Str label(Storage &s, NSString *text)
{
    if (!text)
        return {};
    s.strings.push_back(text.UTF8String ?: "");
    return wire::str(s.strings.back());
}

wire::Size size(MTLSize v)
{
    wire::Size out;
    out.width = v.width;
    out.height = v.height;
    out.depth = v.depth;
    return out;
}

}  // namespace

void BuildComputeDescriptor(WebMetalDevice *device, id self, SEL _cmd, MTLComputePipelineDescriptor *in,
                            ComputeDescriptor &out)
{
    Storage &s = out.storage;
    wire::ComputePipelineDescriptor &d = out.descriptor;
    d.binaryArchives = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "binaryArchives"),
                                                "binaryArchives"));
    d.binaryLibraries = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "binaryLibraries"),
                                                 "binaryLibraries"));
    d.buffers = buffers(s, self, _cmd, get<id>(self, _cmd, in, "buffers"));
    d.computeFunction = FunctionHandle(device, self, _cmd, get<id>(self, _cmd, in, "computeFunction"));
    d.driverCompilerOptions = value(s, self, _cmd, get<id>(self, _cmd, in, "driverCompilerOptions"),
                                    "driverCompilerOptions");
    d.dynamicLibraries = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "dynamicLibraries"),
                                                  "dynamicLibraries"));
    d.forceResourceIndex = get<BOOL>(self, _cmd, in, "forceResourceIndex");
    d.functionPointers = functions(s, device, self, _cmd, get<NSArray *>(self, _cmd, in, "functionPointers"));
    d.gpuCompilerSpiOptions = value(s, self, _cmd, get<id>(self, _cmd, in, "gpuCompilerSPIOptions"),
                                    "gpuCompilerSPIOptions");
    d.insertLibraries = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "insertLibraries"),
                                                 "insertLibraries"));
    d.label = label(s, get<NSString *>(self, _cmd, in, "label"));
    d.linkedFunctions = linked(s, device, self, _cmd, get<id>(self, _cmd, in, "linkedFunctions"));
    d.maxAccelerationStructureTraversalDepth = get<NSUInteger>(self, _cmd, in, "maxAccelerationStructureTraversalDepth");
    d.maxCallStackDepth = get<NSUInteger>(self, _cmd, in, "maxCallStackDepth");
    d.maxTotalThreadsPerThreadgroup = get<NSUInteger>(self, _cmd, in, "maxTotalThreadsPerThreadgroup");
    d.needsCustomBorderColorSamplers = get<BOOL>(self, _cmd, in, "needsCustomBorderColorSamplers");
    d.openClModeEnabled = get<BOOL>(self, _cmd, in, "openCLModeEnabled");
    d.openGlModeEnabled = get<BOOL>(self, _cmd, in, "openGLModeEnabled");
    d.pipelineLibrary = get<id>(self, _cmd, in, "pipelineLibrary") != nil;
    d.pluginData = value(s, self, _cmd, get<id>(self, _cmd, in, "pluginData"), "pluginData");
    d.preloadedLibraries = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "preloadedLibraries"),
                                                    "preloadedLibraries"));
    d.profileControl = profile(self, _cmd, get<id>(self, _cmd, in, "profileControl"));
    d.resourceIndex = get<NSUInteger>(self, _cmd, in, "resourceIndex");
    if (MTLStageInputOutputDescriptor *stage = get<id>(self, _cmd, in, "stageInputDescriptor")) {
        d.stageInputDescriptor.present = true;
        wire::StageInputOutputDescriptor &w = d.stageInputDescriptor.value;
        for (NSUInteger i = 0; i < kAttributeSlots; i++) {
            MTLAttributeDescriptor *a = stage.attributes[i];
            wire::Attribute attribute;
            attribute.bufferIndex = get<NSUInteger>(self, _cmd, a, "bufferIndex");
            attribute.format = get<NSUInteger>(self, _cmd, a, "format");
            attribute.offset = get<NSUInteger>(self, _cmd, a, "offset");
            out.attributes.push_back(attribute);
            MTLBufferLayoutDescriptor *l = stage.layouts[i];
            wire::BufferLayout layout;
            layout.stepFunction = get<NSUInteger>(self, _cmd, l, "stepFunction");
            layout.stepRate = get<NSUInteger>(self, _cmd, l, "stepRate");
            layout.stride = get<NSUInteger>(self, _cmd, l, "stride");
            out.layouts.push_back(layout);
        }
        w.attributes = {out.attributes.data(), out.attributes.size()};
        w.layouts = {out.layouts.data(), out.layouts.size()};
        w.indexBufferIndex = get<NSUInteger>(self, _cmd, stage, "indexBufferIndex");
        w.indexType = get<NSUInteger>(self, _cmd, stage, "indexType");
    }
    d.supportAddingBinaryFunctions = get<BOOL>(self, _cmd, in, "supportAddingBinaryFunctions");
    d.supportIndirectCommandBuffers = get<BOOL>(self, _cmd, in, "supportIndirectCommandBuffers");
    d.textureWriteFpRoundingMode = get<NSInteger>(self, _cmd, in, "textureWriteFPRoundingMode");
    d.textureWriteRoundingMode = get<NSInteger>(self, _cmd, in, "textureWriteRoundingMode");
    d.threadGroupSizeIsMultipleOfThreadExecutionWidth =
        get<BOOL>(self, _cmd, in, "threadGroupSizeIsMultipleOfThreadExecutionWidth");
}

void BuildRenderDescriptor(WebMetalDevice *device, id self, SEL _cmd, MTLRenderPipelineDescriptor *in,
                           RenderDescriptor &out)
{
    Storage &s = out.storage;
    wire::RenderPipelineDescriptor &d = out.descriptor;
    d.alphaTestEnabled = get<BOOL>(self, _cmd, in, "isAlphaTestEnabled");
    d.alphaTestFunction = get<NSUInteger>(self, _cmd, in, "alphaTestFunction");
    d.alphaToCoverageEnabled = get<BOOL>(self, _cmd, in, "isAlphaToCoverageEnabled");
    d.alphaToOneEnabled = get<BOOL>(self, _cmd, in, "isAlphaToOneEnabled");
    d.binaryArchives = handles(s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "binaryArchives"),
                                                "binaryArchives"));
    d.clipDistanceEnableMask = get<uint8_t>(self, _cmd, in, "clipDistanceEnableMask");
    id colors = get<id>(self, _cmd, in, "colorAttachments");
    for (NSUInteger i = 0; i < kColorSlots; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *c =
            [(MTLRenderPipelineColorAttachmentDescriptorArray *)colors objectAtIndexedSubscript:i];
        wire::RenderPipelineColorAttachment w;
        w.alphaBlendOperation = get<NSUInteger>(self, _cmd, c, "alphaBlendOperation");
        w.blendingEnabled = get<BOOL>(self, _cmd, c, "isBlendingEnabled");
        w.destinationAlphaBlendFactor = get<NSUInteger>(self, _cmd, c, "destinationAlphaBlendFactor");
        w.destinationRgbBlendFactor = get<NSUInteger>(self, _cmd, c, "destinationRGBBlendFactor");
        w.pixelFormat = get<NSUInteger>(self, _cmd, c, "pixelFormat");
        w.rgbBlendOperation = get<NSUInteger>(self, _cmd, c, "rgbBlendOperation");
        w.sourceAlphaBlendFactor = get<NSUInteger>(self, _cmd, c, "sourceAlphaBlendFactor");
        w.sourceRgbBlendFactor = get<NSUInteger>(self, _cmd, c, "sourceRGBBlendFactor");
        w.writeMask = get<NSUInteger>(self, _cmd, c, "writeMask");
        out.colors.push_back(w);
    }
    d.colorAttachments = {out.colors.data(), out.colors.size()};
    d.colorSampleCount = get<NSUInteger>(self, _cmd, in, "colorSampleCount");
    d.depthAttachmentPixelFormat = get<NSUInteger>(self, _cmd, in, "depthAttachmentPixelFormat");
    d.depthStencilWriteDisabled = get<BOOL>(self, _cmd, in, "isDepthStencilWriteDisabled");
    d.driverCompilerOptions = value(s, self, _cmd, get<id>(self, _cmd, in, "driverCompilerOptions"),
                                    "driverCompilerOptions");
    d.explicitVisibilityGroupId = get<uint32_t>(self, _cmd, in, "explicitVisibilityGroupID");
    d.forceResourceIndex = get<BOOL>(self, _cmd, in, "forceResourceIndex");
    d.forceSoftwareVertexFetch = get<BOOL>(self, _cmd, in, "forceSoftwareVertexFetch");
    d.fragmentBuffers = buffers(s, self, _cmd, get<id>(self, _cmd, in, "fragmentBuffers"));
    d.fragmentDepthCompareClampMask = get<uint32_t>(self, _cmd, in, "fragmentDepthCompareClampMask");
    d.fragmentFunction = FunctionHandle(device, self, _cmd, get<id>(self, _cmd, in, "fragmentFunction"));
    d.fragmentLinkedFunctions = linked(s, device, self, _cmd, get<id>(self, _cmd, in, "fragmentLinkedFunctions"));
    d.fragmentPreloadedLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "fragmentPreloadedLibraries"), "fragmentPreloadedLibraries"));
    d.gpuCompilerSpiOptions = value(s, self, _cmd, get<id>(self, _cmd, in, "gpuCompilerSPIOptions"),
                                    "gpuCompilerSPIOptions");
    d.inputPrimitiveTopology = get<NSUInteger>(self, _cmd, in, "inputPrimitiveTopology");
    d.insertFragmentLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "insertFragmentLibraries"), "insertFragmentLibraries"));
    d.insertMeshLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "insertMeshLibraries"), "insertMeshLibraries"));
    d.insertObjectLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "insertObjectLibraries"), "insertObjectLibraries"));
    d.insertVertexLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "insertVertexLibraries"), "insertVertexLibraries"));
    d.label = label(s, get<NSString *>(self, _cmd, in, "label"));
    d.logicOperation = get<NSUInteger>(self, _cmd, in, "logicOperation");
    d.logicOperationEnabled = get<BOOL>(self, _cmd, in, "isLogicOperationEnabled");
    d.maxAccelerationStructureTraversalDepth = get<NSUInteger>(self, _cmd, in, "maxAccelerationStructureTraversalDepth");
    d.maxFragmentCallStackDepth = get<NSUInteger>(self, _cmd, in, "maxFragmentCallStackDepth");
    d.maxMeshCallStackDepth = get<NSUInteger>(self, _cmd, in, "maxMeshCallStackDepth");
    d.maxObjectCallStackDepth = get<NSUInteger>(self, _cmd, in, "maxObjectCallStackDepth");
    d.maxPipelineChildren = size(get<MTLSize>(self, _cmd, in, "maxPipelineChildren"));
    d.maxTessellationFactor = get<NSUInteger>(self, _cmd, in, "maxTessellationFactor");
    d.maxTotalThreadsPerMeshThreadgroup = get<NSUInteger>(self, _cmd, in, "maxTotalThreadsPerMeshThreadgroup");
    d.maxTotalThreadsPerObjectThreadgroup = get<NSUInteger>(self, _cmd, in, "maxTotalThreadsPerObjectThreadgroup");
    d.maxVertexAmplificationCount = get<NSUInteger>(self, _cmd, in, "maxVertexAmplificationCount");
    d.maxVertexCallStackDepth = get<NSUInteger>(self, _cmd, in, "maxVertexCallStackDepth");
    d.meshFunction = FunctionHandle(device, self, _cmd, get<id>(self, _cmd, in, "meshFunction"));
    d.meshLinkedFunctions = linked(s, device, self, _cmd, get<id>(self, _cmd, in, "meshLinkedFunctions"));
    d.meshPreloadedLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "meshPreloadedLibraries"), "meshPreloadedLibraries"));
    d.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth =
        get<BOOL>(self, _cmd, in, "meshThreadgroupSizeIsMultipleOfThreadExecutionWidth");
    d.meshThreadsPerThreadgroup = size(get<MTLSize>(self, _cmd, in, "meshThreadsPerThreadgroup"));
    d.needsCustomBorderColorSamplers = get<BOOL>(self, _cmd, in, "needsCustomBorderColorSamplers");
    d.objectFunction = FunctionHandle(device, self, _cmd, get<id>(self, _cmd, in, "objectFunction"));
    d.objectLinkedFunctions = linked(s, device, self, _cmd, get<id>(self, _cmd, in, "objectLinkedFunctions"));
    d.objectPreloadedLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "objectPreloadedLibraries"), "objectPreloadedLibraries"));
    d.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth =
        get<BOOL>(self, _cmd, in, "objectThreadgroupSizeIsMultipleOfThreadExecutionWidth");
    d.objectThreadsPerThreadgroup = size(get<MTLSize>(self, _cmd, in, "objectThreadsPerThreadgroup"));
    d.openGlModeEnabled = get<BOOL>(self, _cmd, in, "openGLModeEnabled");
    d.pipelineLibrary = get<id>(self, _cmd, in, "pipelineLibrary") != nil;
    d.pipelineMemoryLength = get<NSUInteger>(self, _cmd, in, "pipelineMemoryLength");
    d.pluginData = value(s, self, _cmd, get<id>(self, _cmd, in, "pluginData"), "pluginData");
    d.pointCoordLowerLeft = get<BOOL>(self, _cmd, in, "isPointCoordLowerLeft");
    d.pointSizeOutputVs = get<BOOL>(self, _cmd, in, "isPointSizeOutputVS");
    d.pointSmoothEnabled = get<BOOL>(self, _cmd, in, "isPointSmoothEnabled");
    d.postVertexDumpBufferIndex = get<NSUInteger>(self, _cmd, in, "postVertexDumpBufferIndex");
    d.profileControl = profile(self, _cmd, get<id>(self, _cmd, in, "profileControl"));
    d.rasterSampleCount = get<NSUInteger>(self, _cmd, in, "rasterSampleCount");
    d.rasterizationEnabled = get<BOOL>(self, _cmd, in, "isRasterizationEnabled");
    d.resourceIndex = get<NSUInteger>(self, _cmd, in, "resourceIndex");
    d.sampleCount = get<NSUInteger>(self, _cmd, in, "sampleCount");
    d.sampleCoverage = get<float>(self, _cmd, in, "sampleCoverage");
    d.sampleCoverageInvert = get<BOOL>(self, _cmd, in, "sampleCoverageInvert");
    d.sampleMask = get<NSUInteger>(self, _cmd, in, "sampleMask");
    d.stencilAttachmentPixelFormat = get<NSUInteger>(self, _cmd, in, "stencilAttachmentPixelFormat");
    d.supportAddingFragmentBinaryFunctions = get<BOOL>(self, _cmd, in, "supportAddingFragmentBinaryFunctions");
    d.supportAddingMeshBinaryFunctions = get<BOOL>(self, _cmd, in, "supportAddingMeshBinaryFunctions");
    d.supportAddingObjectBinaryFunctions = get<BOOL>(self, _cmd, in, "supportAddingObjectBinaryFunctions");
    d.supportAddingVertexBinaryFunctions = get<BOOL>(self, _cmd, in, "supportAddingVertexBinaryFunctions");
    d.supportIndirectCommandBuffers = get<BOOL>(self, _cmd, in, "supportIndirectCommandBuffers");
    d.tessellationControlPointIndexType = get<NSUInteger>(self, _cmd, in, "tessellationControlPointIndexType");
    d.tessellationFactorFormat = get<NSUInteger>(self, _cmd, in, "tessellationFactorFormat");
    d.tessellationFactorScaleEnabled = get<BOOL>(self, _cmd, in, "isTessellationFactorScaleEnabled");
    d.tessellationFactorStepFunction = get<NSUInteger>(self, _cmd, in, "tessellationFactorStepFunction");
    d.tessellationOutputWindingOrder = get<NSUInteger>(self, _cmd, in, "tessellationOutputWindingOrder");
    d.tessellationPartitionMode = get<NSUInteger>(self, _cmd, in, "tessellationPartitionMode");
    d.textureWriteRoundingMode = get<NSInteger>(self, _cmd, in, "textureWriteRoundingMode");
    d.twoSideEnabled = get<BOOL>(self, _cmd, in, "isTwoSideEnabled");
    d.vertexAmplificationMode = get<NSUInteger>(self, _cmd, in, "vertexAmplificationMode");
    d.vertexBuffers = buffers(s, self, _cmd, get<id>(self, _cmd, in, "vertexBuffers"));
    d.vertexDepthCompareClampMask = get<uint32_t>(self, _cmd, in, "vertexDepthCompareClampMask");
    if (MTLVertexDescriptor *vertex = get<id>(self, _cmd, in, "vertexDescriptor")) {
        d.vertexDescriptor.present = true;
        for (NSUInteger i = 0; i < kAttributeSlots; i++) {
            MTLVertexAttributeDescriptor *a = vertex.attributes[i];
            wire::VertexAttribute attribute;
            attribute.bufferIndex = get<NSUInteger>(self, _cmd, a, "bufferIndex");
            attribute.format = get<NSUInteger>(self, _cmd, a, "format");
            attribute.offset = get<NSUInteger>(self, _cmd, a, "offset");
            out.vertexAttributes.push_back(attribute);
            MTLVertexBufferLayoutDescriptor *l = vertex.layouts[i];
            wire::VertexBufferLayout layout;
            layout.stepFunction = get<NSUInteger>(self, _cmd, l, "stepFunction");
            layout.stepRate = get<NSUInteger>(self, _cmd, l, "stepRate");
            layout.stride = get<NSUInteger>(self, _cmd, l, "stride");
            out.vertexLayouts.push_back(layout);
        }
        d.vertexDescriptor.value.attributes = {out.vertexAttributes.data(), out.vertexAttributes.size()};
        d.vertexDescriptor.value.layouts = {out.vertexLayouts.data(), out.vertexLayouts.size()};
    }
    d.vertexFunction = FunctionHandle(device, self, _cmd, get<id>(self, _cmd, in, "vertexFunction"));
    d.vertexLinkedFunctions = linked(s, device, self, _cmd, get<id>(self, _cmd, in, "vertexLinkedFunctions"));
    d.vertexPreloadedLibraries = handles(
        s, ObjectHandles(self, _cmd, get<NSArray *>(self, _cmd, in, "vertexPreloadedLibraries"), "vertexPreloadedLibraries"));
    d.meshBuffers = buffers(s, self, _cmd, get<id>(self, _cmd, in, "meshBuffers"));
    d.objectBuffers = buffers(s, self, _cmd, get<id>(self, _cmd, in, "objectBuffers"));
}

}  // namespace webmetal
