// Pipeline cases: compute and render pipeline reflection (bindings and the
// older argument objects), asynchronous creation, creation errors,
// descriptor defaults, pipeline state properties, rasterization-disabled
// pipelines, binary archives and dynamic libraries.

#import "WMRefCase.h"
#import "LibrariesSupport.h"

#pragma mark - Reflection text

static NSString *WMDescribeStruct(MTLStructType *structType, NSUInteger depth);

static NSString *WMDescribeArray(MTLArrayType *array, NSUInteger depth) {
    NSMutableString *text = [NSMutableString stringWithFormat:@"array(element=%lu length=%lu stride=%lu index-stride=%lu",
                             (unsigned long)array.elementType, (unsigned long)array.arrayLength,
                             (unsigned long)array.stride, (unsigned long)array.argumentIndexStride];
    if (array.elementStructType) [text appendFormat:@" %@", WMDescribeStruct(array.elementStructType, depth + 1)];
    if (array.elementArrayType) [text appendFormat:@" %@", WMDescribeArray(array.elementArrayType, depth + 1)];
    [text appendString:@")"];
    return text;
}

static NSString *WMDescribePointer(MTLPointerType *pointer) {
    if (!pointer) return @"pointer(none)";
    return [NSString stringWithFormat:@"pointer(element=%lu access=%lu alignment=%lu size=%lu argument-buffer=%d)",
            (unsigned long)pointer.elementType, (unsigned long)pointer.access, (unsigned long)pointer.alignment,
            (unsigned long)pointer.dataSize, pointer.elementIsArgumentBuffer];
}

static NSString *WMDescribeStruct(MTLStructType *structType, NSUInteger depth) {
    if (!structType) return @"struct(none)";
    if (depth > 8) return @"struct(...)";
    NSMutableArray<NSString *> *members = [NSMutableArray array];
    for (MTLStructMember *member in structType.members) {
        NSMutableString *text = [NSMutableString stringWithFormat:@"%@@%lu:%lu index=%lu", member.name,
                                 (unsigned long)member.offset, (unsigned long)member.dataType,
                                 (unsigned long)member.argumentIndex];
        if (member.structType) [text appendFormat:@" %@", WMDescribeStruct(member.structType, depth + 1)];
        if (member.arrayType) [text appendFormat:@" %@", WMDescribeArray(member.arrayType, depth + 1)];
        if (member.pointerType) [text appendFormat:@" %@", WMDescribePointer(member.pointerType)];
        MTLTextureReferenceType *texture = member.textureReferenceType;
        if (texture)
            [text appendFormat:@" texture(type=%lu data=%lu access=%lu depth=%d)", (unsigned long)texture.textureType,
             (unsigned long)texture.textureDataType, (unsigned long)texture.access, texture.isDepthTexture];
        [members addObject:text];
    }
    return [NSString stringWithFormat:@"struct{%@}", [members componentsJoinedByString:@"; "]];
}

NSArray<NSString *> *WMRefDescribeBindings(NSArray<id<MTLBinding>> *bindings) {
    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    for (id<MTLBinding> binding in bindings) {
        NSMutableString *line = [NSMutableString stringWithFormat:@"type=%ld index=%lu %@ access=%lu used=%d argument=%d",
                                 (long)binding.type, (unsigned long)binding.index, binding.name,
                                 (unsigned long)binding.access, binding.used, binding.argument];
        if ([binding conformsToProtocol:@protocol(MTLBufferBinding)]) {
            id<MTLBufferBinding> buffer = (id<MTLBufferBinding>)binding;
            [line appendFormat:@" alignment=%lu size=%lu data=%lu %@ %@", (unsigned long)buffer.bufferAlignment,
             (unsigned long)buffer.bufferDataSize, (unsigned long)buffer.bufferDataType,
             WMDescribePointer(buffer.bufferPointerType), WMDescribeStruct(buffer.bufferStructType, 0)];
        }
        if ([binding conformsToProtocol:@protocol(MTLTextureBinding)]) {
            id<MTLTextureBinding> texture = (id<MTLTextureBinding>)binding;
            [line appendFormat:@" texture-type=%lu data=%lu depth=%d array-length=%lu",
             (unsigned long)texture.textureType, (unsigned long)texture.textureDataType, texture.depthTexture,
             (unsigned long)texture.arrayLength];
        }
        if ([binding conformsToProtocol:@protocol(MTLThreadgroupBinding)]) {
            id<MTLThreadgroupBinding> threadgroup = (id<MTLThreadgroupBinding>)binding;
            [line appendFormat:@" threadgroup-alignment=%lu threadgroup-size=%lu",
             (unsigned long)threadgroup.threadgroupMemoryAlignment,
             (unsigned long)threadgroup.threadgroupMemoryDataSize];
        }
        [lines addObject:line];
    }
    return WMRefSortedNames(lines);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
NSArray<NSString *> *WMRefDescribeArguments(NSArray<MTLArgument *> *arguments) {
    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    for (MTLArgument *argument in arguments) {
        NSMutableString *line = [NSMutableString stringWithFormat:@"type=%lu index=%lu %@ access=%lu active=%d",
                                 (unsigned long)argument.type, (unsigned long)argument.index, argument.name,
                                 (unsigned long)argument.access, argument.active];
        switch (argument.type) {
        case MTLArgumentTypeBuffer:
            [line appendFormat:@" alignment=%lu size=%lu data=%lu %@ %@", (unsigned long)argument.bufferAlignment,
             (unsigned long)argument.bufferDataSize, (unsigned long)argument.bufferDataType,
             WMDescribePointer(argument.bufferPointerType), WMDescribeStruct(argument.bufferStructType, 0)];
            break;
        case MTLArgumentTypeTexture:
            [line appendFormat:@" texture-type=%lu data=%lu depth=%d array-length=%lu",
             (unsigned long)argument.textureType, (unsigned long)argument.textureDataType, argument.isDepthTexture,
             (unsigned long)argument.arrayLength];
            break;
        case MTLArgumentTypeThreadgroupMemory:
            [line appendFormat:@" threadgroup-alignment=%lu threadgroup-size=%lu",
             (unsigned long)argument.threadgroupMemoryAlignment, (unsigned long)argument.threadgroupMemoryDataSize];
            break;
        default:
            break;
        }
        [lines addObject:line];
    }
    return WMRefSortedNames(lines);
}
#pragma clang diagnostic pop

#pragma mark - Descriptors

static MTLRenderPipelineDescriptor *WMReflectRenderDescriptor(WMCase *c) {
    MTLVertexDescriptor *vertex = [MTLVertexDescriptor vertexDescriptor];
    vertex.attributes[0].format = MTLVertexFormatFloat4;
    vertex.attributes[0].offset = 0;
    vertex.attributes[0].bufferIndex = 0;
    vertex.attributes[1].format = MTLVertexFormatHalf2;
    vertex.attributes[1].offset = 16;
    vertex.attributes[1].bufferIndex = 0;
    vertex.attributes[3].format = MTLVertexFormatUChar4;
    vertex.attributes[3].offset = 20;
    vertex.attributes[3].bufferIndex = 0;
    vertex.attributes[4].format = MTLVertexFormatFloat;
    vertex.attributes[4].offset = 24;
    vertex.attributes[4].bufferIndex = 0;
    vertex.layouts[0].stride = 28;
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"pip_reflect_vertex"];
    descriptor.fragmentFunction = [c function:@"pip_reflect_fragment"];
    descriptor.vertexDescriptor = vertex;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA32Uint;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
    return descriptor;
}

static MTLRenderPipelineDescriptor *WMSimpleRenderDescriptor(WMCase *c) {
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"pip_simple_vertex"];
    descriptor.fragmentFunction = [c function:@"pip_simple_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    return descriptor;
}

// MTLPipelineOptionArgumentInfo (renamed MTLPipelineOptionBindingInfo in
// macOS 13 and iOS 16, same value) and MTLPipelineOptionBufferTypeInfo.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static const MTLPipelineOption WMReflectionOptions = MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo;
#pragma clang diagnostic pop

#pragma mark - Reflection cases

WMREF_CASE(pipelineComputeReflection, "pipeline.compute-reflection", "pipeline.reflection",
           "Compute pipeline reflection with argument and buffer type information")
{
    MTLComputePipelineReflection *reflection = nil;
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [c.device newComputePipelineStateWithFunction:[c function:@"pip_reflect_compute"]
                                              options:WMReflectionOptions reflection:&reflection error:&error];
    if (!state) [c fail:@"reflection pipeline: %@", error];
    [c value:@"reflection-nil" boolean:reflection == nil rule:WMExact()];
    if (@available(macOS 13.0, iOS 16.0, *))
        [c value:@"bindings" strings:WMRefDescribeBindings(reflection.bindings) rule:WMExact()];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [c value:@"arguments" strings:WMRefDescribeArguments(reflection.arguments) rule:WMExact()];
#pragma clang diagnostic pop
    [c value:@"static-threadgroup-memory" u64:state.staticThreadgroupMemoryLength rule:WMDevice()];
}

WMREF_CASE(pipelineRenderReflection, "pipeline.render-reflection", "pipeline.reflection",
           "Render pipeline reflection for the vertex and fragment stages")
{
    MTLRenderPipelineReflection *reflection = nil;
    NSError *error = nil;
    id<MTLRenderPipelineState> state =
        [c.device newRenderPipelineStateWithDescriptor:WMReflectRenderDescriptor(c) options:WMReflectionOptions
                                            reflection:&reflection error:&error];
    if (!state) [c fail:@"reflection render pipeline: %@", error];
    if (@available(macOS 13.0, iOS 16.0, *)) {
        [c value:@"vertex-bindings" strings:WMRefDescribeBindings(reflection.vertexBindings) rule:WMExact()];
        [c value:@"fragment-bindings" strings:WMRefDescribeBindings(reflection.fragmentBindings) rule:WMExact()];
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [c value:@"vertex-arguments" strings:WMRefDescribeArguments(reflection.vertexArguments ?: @[]) rule:WMExact()];
    [c value:@"fragment-arguments" strings:WMRefDescribeArguments(reflection.fragmentArguments ?: @[])
        rule:WMExact()];
#pragma clang diagnostic pop
}

WMREF_CASE(pipelineComputeAsync, "pipeline.compute-async", "pipeline.async",
           "newComputePipelineStateWithFunction:options:completionHandler: with reflection")
{
    __block NSUInteger calls = 0;
    __block id<MTLComputePipelineState> state = nil;
    __block MTLComputePipelineReflection *reflection = nil;
    __block NSError *error = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [c.device newComputePipelineStateWithFunction:[c function:@"pip_reflect_compute"] options:WMReflectionOptions
                                completionHandler:^(id<MTLComputePipelineState> s, MTLComputePipelineReflection *r,
                                                    NSError *e) {
        calls++;
        state = s;
        reflection = r;
        error = e;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"handler-calls" u64:calls rule:WMExact()];
    [c value:@"state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"error" value:WMRefStableError(error) rule:WMExact()];
    if (@available(macOS 13.0, iOS 16.0, *))
        [c value:@"bindings" strings:WMRefDescribeBindings(reflection.bindings ?: @[]) rule:WMExact()];

    // Without reflection options the handler form without reflection.
    calls = 0;
    state = nil;
    [c.device newComputePipelineStateWithFunction:[c function:@"pip_marker"]
                                completionHandler:^(id<MTLComputePipelineState> s, NSError *e) {
        calls++;
        state = s;
        error = e;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"plain.handler-calls" u64:calls rule:WMExact()];
    [c value:@"plain.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"plain.error" value:WMRefStableError(error) rule:WMExact()];
}

WMREF_CASE(pipelineRenderAsync, "pipeline.render-async", "pipeline.async",
           "newRenderPipelineStateWithDescriptor:completionHandler: for a valid and an unlinkable pipeline")
{
    __block NSUInteger calls = 0;
    __block id<MTLRenderPipelineState> state = nil;
    __block NSError *error = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [c.device newRenderPipelineStateWithDescriptor:WMSimpleRenderDescriptor(c)
                                 completionHandler:^(id<MTLRenderPipelineState> s, NSError *e) {
        calls++;
        state = s;
        error = e;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"valid.handler-calls" u64:calls rule:WMExact()];
    [c value:@"valid.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"valid.error" value:WMRefStableError(error) rule:WMExact()];

    calls = 0;
    state = nil;
    error = nil;
    MTLRenderPipelineDescriptor *unmatched = WMSimpleRenderDescriptor(c);
    unmatched.fragmentFunction = [c function:@"pip_unmatched_fragment"];
    [c.device newRenderPipelineStateWithDescriptor:unmatched
                                 completionHandler:^(id<MTLRenderPipelineState> s, NSError *e) {
        calls++;
        state = s;
        error = e;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"unmatched.handler-calls" u64:calls rule:WMExact()];
    [c value:@"unmatched.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"unmatched.error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
}

WMREF_CASE(pipelineRenderErrors, "pipeline.render-errors", "pipeline.errors",
           "Render pipeline creation failures that return an NSError")
{
    // The fragment function reads a varying the vertex function does not
    // write.
    MTLRenderPipelineDescriptor *unmatched = WMSimpleRenderDescriptor(c);
    unmatched.fragmentFunction = [c function:@"pip_unmatched_fragment"];
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [c.device newRenderPipelineStateWithDescriptor:unmatched error:&error];
    [c value:@"unmatched-varying.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"unmatched-varying.error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];

    // A vertex function with [[stage_in]] and no vertex descriptor.
    MTLRenderPipelineDescriptor *noDescriptor = WMSimpleRenderDescriptor(c);
    noDescriptor.vertexFunction = [c function:@"pip_stage_vertex"];
    noDescriptor.fragmentFunction = nil;
    error = nil;
    state = [c.device newRenderPipelineStateWithDescriptor:noDescriptor error:&error];
    [c value:@"stage-in-without-descriptor.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"stage-in-without-descriptor.error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
}

WMREF_CASE(pipelineDescriptorDefaults, "pipeline.descriptor-defaults", "pipeline.descriptors",
           "Default values of render and compute pipeline descriptors")
{
    MTLRenderPipelineDescriptor *render = [MTLRenderPipelineDescriptor new];
    // Stated in MTLRenderPipeline.h.
    MTLRenderPipelineColorAttachmentDescriptor *attachment = render.colorAttachments[0];
    [c value:@"color.pixel-format" u64:attachment.pixelFormat rule:WMExact()];
    [c value:@"color.blending-enabled" boolean:attachment.blendingEnabled rule:WMExact()];
    [c value:@"color.source-rgb" u64:attachment.sourceRGBBlendFactor rule:WMExact()];
    [c value:@"color.destination-rgb" u64:attachment.destinationRGBBlendFactor rule:WMExact()];
    [c value:@"color.rgb-operation" u64:attachment.rgbBlendOperation rule:WMExact()];
    [c value:@"color.source-alpha" u64:attachment.sourceAlphaBlendFactor rule:WMExact()];
    [c value:@"color.destination-alpha" u64:attachment.destinationAlphaBlendFactor rule:WMExact()];
    [c value:@"color.alpha-operation" u64:attachment.alphaBlendOperation rule:WMExact()];
    [c value:@"color.write-mask" u64:attachment.writeMask rule:WMExact()];
    [c value:@"max-call-stack-depth" u64:render.maxVertexCallStackDepth rule:WMExact()];
    [c value:@"depth-format" u64:render.depthAttachmentPixelFormat rule:WMExact()];
    [c value:@"stencil-format" u64:render.stencilAttachmentPixelFormat rule:WMExact()];
    // Not stated in the header; recorded for reference.
    WMRule *unstated = WMNone(@"default not stated in MTLRenderPipeline.h");
    [c value:@"raster-sample-count" u64:render.rasterSampleCount rule:unstated];
    [c value:@"alpha-to-coverage" boolean:render.alphaToCoverageEnabled rule:unstated];
    [c value:@"alpha-to-one" boolean:render.alphaToOneEnabled rule:unstated];
    [c value:@"rasterization-enabled" boolean:render.rasterizationEnabled rule:unstated];
    [c value:@"input-primitive-topology" u64:render.inputPrimitiveTopology rule:unstated];
    [c value:@"max-tessellation-factor" u64:render.maxTessellationFactor rule:unstated];
    [c value:@"tessellation-partition-mode" u64:render.tessellationPartitionMode rule:unstated];
    [c value:@"tessellation-factor-format" u64:render.tessellationFactorFormat rule:unstated];
    [c value:@"tessellation-step-function" u64:render.tessellationFactorStepFunction rule:unstated];
    [c value:@"tessellation-winding" u64:render.tessellationOutputWindingOrder rule:unstated];
    [c value:@"tessellation-control-point-index" u64:render.tessellationControlPointIndexType rule:unstated];
    [c value:@"support-indirect-command-buffers" boolean:render.supportIndirectCommandBuffers rule:unstated];
    [c value:@"max-vertex-amplification-count" u64:render.maxVertexAmplificationCount rule:unstated];

    MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
    [c value:@"compute.max-call-stack-depth" u64:compute.maxCallStackDepth rule:WMExact()];
    [c value:@"compute.max-total-threads" u64:compute.maxTotalThreadsPerThreadgroup rule:unstated];
    [c value:@"compute.multiple-of-execution-width" boolean:compute.threadGroupSizeIsMultipleOfThreadExecutionWidth
        rule:unstated];
    [c value:@"compute.support-indirect-command-buffers" boolean:compute.supportIndirectCommandBuffers
        rule:unstated];
}

WMREF_CASE(pipelineStateProperties, "pipeline.state-properties", "pipeline.descriptors",
           "Pipeline state label, device and indirect-command-buffer support read back from created states")
{
    MTLRenderPipelineDescriptor *render = WMSimpleRenderDescriptor(c);
    render.label = @"wmref render";
    render.supportIndirectCommandBuffers = YES;
    id<MTLRenderPipelineState> renderState = [c renderPipeline:render];
    [c value:@"render.label" string:renderState.label rule:WMExact()];
    [c value:@"render.same-device" boolean:renderState.device == c.device rule:WMExact()];
    [c value:@"render.support-indirect-command-buffers" boolean:renderState.supportIndirectCommandBuffers
        rule:WMExact()];

    MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
    compute.computeFunction = [c function:@"pip_marker"];
    compute.label = @"wmref compute";
    compute.supportIndirectCommandBuffers = YES;
    NSError *error = nil;
    id<MTLComputePipelineState> computeState =
        [c.device newComputePipelineStateWithDescriptor:compute options:MTLPipelineOptionNone reflection:nil
                                                  error:&error];
    if (!computeState) [c fail:@"compute pipeline: %@", error];
    [c value:@"compute.label" string:computeState.label rule:WMExact()];
    [c value:@"compute.same-device" boolean:computeState.device == c.device rule:WMExact()];
    [c value:@"compute.support-indirect-command-buffers" boolean:computeState.supportIndirectCommandBuffers
        rule:WMExact()];
}

WMREF_CASE(pipelineRasterizationDisabled, "pipeline.rasterization-disabled", "pipeline.descriptors",
           "A render pipeline with rasterization disabled whose vertex function writes a buffer")
{
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"pip_void_vertex"];
    descriptor.rasterizationEnabled = NO;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
    id<MTLRenderPipelineState> state = [c renderPipeline:descriptor];
    id<MTLBuffer> out = [c buffer:16 * sizeof(uint32_t)];
    MTLTextureDescriptor *target = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:4 height:4 mipmapped:NO];
    target.usage = MTLTextureUsageRenderTarget;
    target.storageMode = MTLStorageModePrivate;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = [c texture:target];
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    [c render:pass body:^(id<MTLRenderCommandEncoder> e) {
        [e setRenderPipelineState:state];
        [e setVertexBuffer:out offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:2 vertexCount:10];
    }];
    // Vertices 2..11 write 2 * id + 1; the others keep the CPU clear.
    [c data:@"vertex-writes" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

#pragma mark - Binary archives

static NSURL *WMTemporaryURL(NSString *name) {
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
                      [NSString stringWithFormat:@"wmref-%d-%@", getpid(), name]];
    return [NSURL fileURLWithPath:path];
}

WMREF_CASE(archiveRoundTrip, "archive.round-trip", "pipeline.binary-archive",
           "MTLBinaryArchive: add a compute pipeline, serialize, reload and build with failOnBinaryArchiveMiss")
{
    NSError *error = nil;
    MTLBinaryArchiveDescriptor *empty = [MTLBinaryArchiveDescriptor new];
    id<MTLBinaryArchive> archive = [c.device newBinaryArchiveWithDescriptor:empty error:&error];
    [c value:@"create.archive-nil" boolean:archive == nil rule:WMExact()];
    [c error:@"create.error" value:WMRefStableError(error) rule:WMExact()];
    if (!archive) [c fail:@"archive creation failed: %@", error];
    [c value:@"create.same-device" boolean:archive.device == c.device rule:WMExact()];

    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = [c function:@"pip_marker"];
    error = nil;
    BOOL added = [archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error];
    [c value:@"add.result" boolean:added rule:WMExact()];
    [c error:@"add.error" value:WMRefStableError(error) rule:WMExact()];

    NSURL *url = WMTemporaryURL(@"archive.bin");
    error = nil;
    BOOL serialized = [archive serializeToURL:url error:&error];
    [c value:@"serialize.result" boolean:serialized rule:WMExact()];
    [c error:@"serialize.error" value:WMRefStableError(error) rule:WMExact()];
    NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
    [c value:@"serialize.file-nonempty" boolean:[attributes[NSFileSize] unsignedLongLongValue] > 0 rule:WMExact()];
    // The archive size varies between runs of the same program and is not
    // recorded.

    MTLBinaryArchiveDescriptor *load = [MTLBinaryArchiveDescriptor new];
    load.url = url;
    error = nil;
    id<MTLBinaryArchive> loaded = [c.device newBinaryArchiveWithDescriptor:load error:&error];
    [c value:@"load.archive-nil" boolean:loaded == nil rule:WMExact()];
    [c error:@"load.error" value:WMRefStableError(error) rule:WMExact()];

    // A pipeline that the loaded archive contains, with misses failing.
    MTLComputePipelineDescriptor *hit = [MTLComputePipelineDescriptor new];
    hit.computeFunction = [c function:@"pip_marker"];
    hit.binaryArchives = loaded ? @[loaded] : @[];
    error = nil;
    id<MTLComputePipelineState> state =
        [c.device newComputePipelineStateWithDescriptor:hit options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                             reflection:nil error:&error];
    [c value:@"hit.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"hit.error" value:WMRefStableError(error) rule:WMExact()];
    if (state) {
        id<MTLComputePipelineState> hitState = state;
        id<MTLBuffer> out = [c buffer:4 * sizeof(uint32_t)];
        [c compute:^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:hitState];
            [e setBuffer:out offset:0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        }];
        [c data:@"hit.marker" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
    }

    // A pipeline missing from an empty archive, with misses failing.
    id<MTLBinaryArchive> emptyArchive = [c.device newBinaryArchiveWithDescriptor:empty error:NULL];
    MTLComputePipelineDescriptor *miss = [MTLComputePipelineDescriptor new];
    miss.computeFunction = [c function:@"pip_reflect_compute"];
    miss.binaryArchives = emptyArchive ? @[emptyArchive] : @[];
    error = nil;
    state = [c.device newComputePipelineStateWithDescriptor:miss options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                 reflection:nil error:&error];
    [c value:@"miss.state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"miss.error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
    [NSFileManager.defaultManager removeItemAtURL:url error:NULL];
}

WMREF_CASE(archiveInvalidFile, "archive.invalid-file", "pipeline.binary-archive",
           "newBinaryArchiveWithDescriptor:error: for a missing file and for a file that is not an archive")
{
    MTLBinaryArchiveDescriptor *descriptor = [MTLBinaryArchiveDescriptor new];
    descriptor.url = [NSURL fileURLWithPath:@"/nonexistent/wmref-archive.bin"];
    NSError *error = nil;
    id<MTLBinaryArchive> archive = [c.device newBinaryArchiveWithDescriptor:descriptor error:&error];
    [c value:@"missing.archive-nil" boolean:archive == nil rule:WMExact()];
    [c error:@"missing.error" value:WMRefStableError(error) rule:WMExact()];

    uint8_t garbage[512];
    WMPattern(garbage, sizeof garbage, 61);
    NSURL *url = WMTemporaryURL(@"garbage.bin");
    [[NSData dataWithBytes:garbage length:sizeof garbage] writeToURL:url atomically:YES];
    descriptor.url = url;
    error = nil;
    archive = [c.device newBinaryArchiveWithDescriptor:descriptor error:&error];
    [c value:@"garbage.archive-nil" boolean:archive == nil rule:WMExact()];
    [c error:@"garbage.error" value:WMRefStableError(error) rule:WMExact()];
    [NSFileManager.defaultManager removeItemAtURL:url error:NULL];
}

#pragma mark - Dynamic libraries

static NSString *const WMDynamicSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "uint wmref_dynamic_mix(uint x) { return x * 2654435761u + 12345u; }\n";

static NSString *const WMDynamicClientSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "extern uint wmref_dynamic_mix(uint x);\n"
     "kernel void src_dynamic_client(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]])\n"
     "{\n"
     "    out[i] = wmref_dynamic_mix(i);\n"
     "}\n";

WMREF_CASE(libraryDynamic, "library.dynamic", "library.dynamic",
           "MTLDynamicLibrary compiled from source, linked by an executable library and used by a pipeline")
{
    [c require:@"supports-dynamic-libraries" value:c.device.supportsDynamicLibraries];
    NSURL *url = WMTemporaryURL(@"dynamic.metallib");
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.libraryType = MTLLibraryTypeDynamic;
    options.installName = url.path;
    NSError *error = nil;
    id<MTLLibrary> source = [c.device newLibraryWithSource:WMDynamicSource options:options error:&error];
    if (!source) [c fail:@"dynamic source library: %@", error];
    [c value:@"source.type" i64:source.type rule:WMExact()];
    [c value:@"source.install-name-matches" boolean:[source.installName isEqualToString:url.path] rule:WMExact()];
    id<MTLDynamicLibrary> dynamic = [c.device newDynamicLibrary:source error:&error];
    [c value:@"dynamic.nil" boolean:dynamic == nil rule:WMExact()];
    [c error:@"dynamic.error" value:WMRefStableError(error) rule:WMExact()];
    if (!dynamic) [c fail:@"newDynamicLibrary failed: %@", error];
    error = nil;
    BOOL serialized = [dynamic serializeToURL:url error:&error];
    [c value:@"serialize.result" boolean:serialized rule:WMExact()];
    [c error:@"serialize.error" value:WMRefStableError(error) rule:WMExact()];

    MTLCompileOptions *clientOptions = [MTLCompileOptions new];
    clientOptions.libraries = @[dynamic];
    id<MTLLibrary> client = [c.device newLibraryWithSource:WMDynamicClientSource options:clientOptions error:&error];
    if (!client) [c fail:@"client library: %@", error];
    NSData *out = WMRefRunWordsKernel(c, [client newFunctionWithName:@"src_dynamic_client"], 8, 8);
    [c data:@"client.results" elem:WMElemU32 bytes:out.bytes length:out.length rule:WMExact()];

    // An executable library cannot become a dynamic library.
    id<MTLLibrary> executable = [c.device newLibraryWithSource:WMDynamicClientSource options:clientOptions
                                                         error:NULL];
    error = nil;
    id<MTLDynamicLibrary> wrong = executable ? [c.device newDynamicLibrary:executable error:&error] : nil;
    [c value:@"from-executable.nil" boolean:wrong == nil rule:WMExact()];
    [c error:@"from-executable.error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];

    // Loading the serialized library from its URL.
    error = nil;
    id<MTLDynamicLibrary> loaded = [c.device newDynamicLibraryWithURL:url error:&error];
    [c value:@"load.nil" boolean:loaded == nil rule:WMExact()];
    [c error:@"load.error" value:WMRefStableError(error) rule:WMExact()];
    [c value:@"load.install-name-matches" boolean:[loaded.installName isEqualToString:url.path] rule:WMExact()];
    [NSFileManager.defaultManager removeItemAtURL:url error:NULL];
}

#pragma mark - Stitched libraries

static NSString *const WMStitchableSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "[[stitchable]] int wmref_add(int a, int b) { return a + b; }\n"
     "[[stitchable]] int wmref_mul(int a, int b) { return a * b; }\n";

static NSString *const WMStitchedCallerSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "using StitchedFunction = int(int, int, int);\n"
     "kernel void src_call_stitched(visible_function_table<StitchedFunction> table [[buffer(0)]],\n"
     "                              device int *out [[buffer(1)]], uint i [[thread_position_in_grid]])\n"
     "{\n"
     "    out[i] = table[0](int(i), 3, 5);\n"
     "}\n";

WMREF_CASE(libraryStitched, "library.stitched", "library.stitched",
           "newLibraryWithStitchedDescriptor:error: building (a + b) * c from stitchable functions, called through a visible function table")
{
    [c require:@"supports-function-pointers" value:c.device.supportsFunctionPointers];
    NSError *error = nil;
    id<MTLLibrary> stitchable = [c.device newLibraryWithSource:WMStitchableSource options:nil error:&error];
    if (!stitchable) [c fail:@"stitchable library: %@", error];
    id<MTLFunction> add = [stitchable newFunctionWithName:@"wmref_add"];
    id<MTLFunction> mul = [stitchable newFunctionWithName:@"wmref_mul"];
    if (!add || !mul) [c fail:@"stitchable functions missing"];
    [c value:@"stitchable.type" i64:add.functionType rule:WMExact()];

    MTLFunctionStitchingInputNode *a = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
    MTLFunctionStitchingInputNode *b = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1];
    MTLFunctionStitchingInputNode *k = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:2];
    MTLFunctionStitchingFunctionNode *sum =
        [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"wmref_add" arguments:@[a, b] controlDependencies:@[]];
    MTLFunctionStitchingFunctionNode *product =
        [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"wmref_mul" arguments:@[sum, k]
                                           controlDependencies:@[]];
    MTLFunctionStitchingGraph *graph = [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"wmref_stitched"
                                                                                         nodes:@[sum, product]
                                                                                    outputNode:product
                                                                                    attributes:@[]];
    MTLStitchedLibraryDescriptor *descriptor = [MTLStitchedLibraryDescriptor new];
    descriptor.functions = @[add, mul];
    descriptor.functionGraphs = @[graph];
    error = nil;
    id<MTLLibrary> stitched = [c.device newLibraryWithStitchedDescriptor:descriptor error:&error];
    [c value:@"stitched.library-nil" boolean:stitched == nil rule:WMExact()];
    [c error:@"stitched.error" value:WMRefStableError(error) rule:WMExact()];
    if (!stitched) [c fail:@"stitched library: %@", error];
    [c value:@"stitched.function-names" strings:WMRefSortedNames(stitched.functionNames) rule:WMExact()];
    id<MTLFunction> function = [stitched newFunctionWithName:@"wmref_stitched"];
    if (!function) [c fail:@"stitched function missing"];
    [c value:@"stitched.type" i64:function.functionType rule:WMExact()];

    id<MTLLibrary> callerLibrary = [c.device newLibraryWithSource:WMStitchedCallerSource options:nil error:&error];
    if (!callerLibrary) [c fail:@"caller library: %@", error];
    MTLComputePipelineDescriptor *pipelineDescriptor = [MTLComputePipelineDescriptor new];
    pipelineDescriptor.computeFunction = [callerLibrary newFunctionWithName:@"src_call_stitched"];
    MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
    linked.functions = @[function];
    pipelineDescriptor.linkedFunctions = linked;
    id<MTLComputePipelineState> pipeline =
        [c.device newComputePipelineStateWithDescriptor:pipelineDescriptor options:MTLPipelineOptionNone
                                             reflection:nil error:&error];
    if (!pipeline) [c fail:@"caller pipeline: %@", error];
    MTLVisibleFunctionTableDescriptor *tableDescriptor = [MTLVisibleFunctionTableDescriptor visibleFunctionTableDescriptor];
    tableDescriptor.functionCount = 1;
    id<MTLVisibleFunctionTable> table = [pipeline newVisibleFunctionTableWithDescriptor:tableDescriptor];
    if (!table) [c fail:@"newVisibleFunctionTableWithDescriptor returned nil"];
    [table setFunction:[pipeline functionHandleWithFunction:function] atIndex:0];
    id<MTLBuffer> out = [c buffer:8 * sizeof(int32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setVisibleFunctionTable:table atBufferIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
    }];
    // (i + 3) * 5 for i = 0..7.
    [c data:@"results" elem:WMElemI32 bytes:out.contents length:out.length rule:WMExact()];
}
