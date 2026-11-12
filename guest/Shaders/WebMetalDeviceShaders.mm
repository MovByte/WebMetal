// The device's shader methods: AGXA10FamilyDevice's libraries, pipelines, binary archives, argument encoders, dynamic
// libraries and indirect command buffers, as a category of WebMetalDevice.

#import "WebMetalPipelineDescriptors.h"
#import "WebMetalShaders.h"
#import "../Commands/WebMetalCommands.h"

#import <objc/message.h>

#include <dlfcn.h>

// The class adopts the protocols that declare these methods; this category, not the class, implements them.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace {

using namespace webmetal;

// MTLPipelineOptionArgumentInfo and MTLPipelineOptionBufferTypeInfo (MTLDevice.h): the reflection Metal returns has
// arguments and their buffers' types.
constexpr MTLPipelineOption kReflectionOptions = MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo;

// -[MTLRenderPipelineDescriptorInternal newSerializedVertexDataWithFlags:options:error:]'s flags as the driver passes
// them (createRenderPipeline, AGXMetalA10 0x21c8e0338): the A10X's compiler flags, which setupCompiler: also gives
// its MTLCompiler (AGXMetalA10 0x21c8d9e8c).
constexpr uint64_t kCompilerFlags = 0x21c;

// The options the driver compiles with: the caller's, plus the compile-statistics option (0x400000) when
// Metal.framework's _MTLCompilePerformanceStatisticsEnabled answers YES (createComputePipeline and
// createRenderPipeline, AGXMetalA10 0x21c8decb4 and 0x21c8e0338). The statistics come from the back end's reply, which
// the host does not produce, so that case is LOUD.
MTLPipelineOption compileOptions(id self, SEL _cmd, MTLPipelineOption options)
{
    using Enabled = BOOL (*)(void);
    static Enabled enabled = (Enabled)dlsym(RTLD_DEFAULT, "_MTLCompilePerformanceStatisticsEnabled");
    if (!enabled)
        WMLoudMethod(self, _cmd, "the running Metal.framework has no _MTLCompilePerformanceStatisticsEnabled");
    if (enabled()) {
        WMLoudMethod(self, _cmd, "Metal.framework records compile performance statistics, which the A10X driver takes "
                                 "from its back-end compiler and the host does not produce");
    }
    return options;
}

// What a pipeline creation reports: the caller's reflection pointer or completion handler decides whether reflection is
// built (the driver builds it whenever one of them is given, createComputePipeline's completion block at
// AGXMetalA10 0x21c8df1bc).
struct Want {
    bool reflection;
    MTLPipelineOption options;
};

// The reflection the caller asked for: built from the host's reply when the options have both reflection bits. Which
// reflection Metal returns for other options is not established, so that case is LOUD.
void checkReflectionOptions(id self, SEL _cmd, const Want &want)
{
    if (want.reflection && (want.options & kReflectionOptions) != kReflectionOptions)
        WMLoudMethod(self, _cmd, "reflection requested with options that lack MTLPipelineOptionArgumentInfo or "
                                 "MTLPipelineOptionBufferTypeInfo, whose effect on the A10X's reflection is not "
                                 "established");
}

// The front of the driver's pipeline creation (createComputePipeline, AGXMetalA10 0x21c8decb4; createRenderPipeline,
// 0x21c8e0338): a nil descriptor, or one Metal.framework's validateWithDevice:error: rejects, fails with an error in
// the driver's domain, code 2, whose description is the validator's. Returns the error, or nil when the descriptor
// passes.
NSError *validate(WebMetalDevice *self, id descriptor)
{
    if (!descriptor)
        return DriverError(2, @"pipeline descriptor is nil");
    NSError *reason = nil;
    using Validate = BOOL (*)(id, SEL, id, NSError **);
    if (((Validate)objc_msgSend)(descriptor, sel_registerName("validateWithDevice:error:"), self, &reason))
        return nil;
    return DriverError(2, reason.localizedDescription);
}

// The queue the plugin runs pipeline completion handlers on. The driver runs them where its back-end compile
// completes, a queue of Metal.framework's compiler connection, never the caller's thread.
dispatch_queue_t completionQueue()
{
    static dispatch_queue_t queue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        queue = dispatch_queue_create("com.apple.WebMetal.pipeline-completion", DISPATCH_QUEUE_SERIAL);
    });
    return queue;
}

// A compute pipeline the host created: the plugin's object with its handle, and its reflection when asked for.
struct ComputeResult {
    WebMetalComputePipelineState *state = nil;
    MTLComputePipelineReflection *reflection = nil;
    NSError *error = nil;
};

template <class Request, class ReplyFields>
ComputeResult createCompute(WebMetalDevice *self, SEL _cmd, MTLComputePipelineDescriptor *descriptor, Request &request,
                            const Want &want, ComputeDescriptor &wire_)
{
    ComputeResult out;
    Connection *connection = DeviceConnection(self);
    request.pipeline = connection->NewHandle();
    request.descriptor = wire_.descriptor;
    wire::Reply<ReplyFields> reply;
    connection->Call(self, _cmd, request, &reply);
    if (!reply.ok) {
        out.error = MetalError(reply.failure);
        return out;
    }
    out.state = [[WebMetalComputePipelineState alloc] initWithDevice:self pipelineStateDescriptor:descriptor];
    AdoptPipelineHandle(out.state, connection, request.pipeline);
    if (want.reflection) {
        if (!reply.value.reflection)
            WMLoudMethod(self, _cmd, "the host returned no reflection for a pipeline whose reflection was asked for");
        out.reflection = NewComputeReflection(self, _cmd, *reply.value.reflection);
    }
    return out;
}

struct RenderResult {
    WebMetalRenderPipelineState *state = nil;
    MTLRenderPipelineReflection *reflection = nil;
    NSError *error = nil;
};

template <class Request, class ReplyFields>
RenderResult createRender(WebMetalDevice *self, SEL _cmd, MTLRenderPipelineDescriptor *descriptor, Request &request,
                          const Want &want, RenderDescriptor &wire_, bool replyHasReflection)
{
    RenderResult out;
    Connection *connection = DeviceConnection(self);
    request.pipeline = connection->NewHandle();
    request.descriptor = wire_.descriptor;
    wire::Reply<ReplyFields> reply;
    connection->Call(self, _cmd, request, &reply);
    if (!reply.ok) {
        out.error = MetalError(reply.failure);
        return out;
    }
    out.state = [[WebMetalRenderPipelineState alloc] initWithDevice:self pipelineStateDescriptor:descriptor];
    AdoptPipelineHandle(out.state, connection, request.pipeline);
    if (want.reflection) {
        if constexpr (std::is_same_v<ReplyFields, wire::CreateRenderPipelineReflectionReply> ||
                      std::is_same_v<ReplyFields, wire::CreateRenderPipelineReflectionAsyncReply>) {
            if (!reply.value.reflection)
                WMLoudMethod(self, _cmd, "the host returned no reflection for a pipeline whose reflection was asked "
                                         "for");
            out.reflection = NewRenderReflection(self, _cmd, *reply.value.reflection, descriptor.vertexDescriptor);
        } else {
            WMLoudMethod(self, _cmd, "the record carries no reflection");
        }
    }
    (void)replyHasReflection;
    return out;
}

// The compute pipeline creation of every form (AGX::UserCommonShaderFactory::createComputePipeline, AGXMetalA10
// 0x21c8decb4). Synchronous forms return the pipeline and fill `reflection` and `error`; asynchronous forms return nil
// and run `handler` with the result.
id<MTLComputePipelineState> newComputePipeline(WebMetalDevice *self, SEL _cmd, MTLComputePipelineDescriptor *descriptor,
                                               MTLPipelineOption options, MTLAutoreleasedComputePipelineReflection *reflection,
                                               NSError **error,
                                               MTLNewComputePipelineStateWithReflectionCompletionHandler handler,
                                               bool handlerTakesReflection)
{
    RequireShaderEntryPoints(self, _cmd);
    options = compileOptions(self, _cmd, options);
    if (error)
        *error = nil;
    if (NSError *invalid = validate(self, descriptor)) {
        if (handler) {
            handler(nil, nil, invalid);
            return nil;
        }
        if (error)
            *error = invalid;
        return nil;
    }
    if (handler)
        descriptor = [descriptor copy];
    // Metal.framework resolves the function's dynamic libraries; on the A10X (supportsDynamicLibraries NO) a function
    // that needs one fails here. As in the driver, the asynchronous forms pass no error pointer and their handler is
    // not called when this fails.
    if (id<MTLFunction> function = descriptor.computeFunction) {
        if (![self loadDynamicLibrariesForFunction:function insertLibraries:descriptor.preloadedLibraries
                                             error:handler ? nullptr : error])
            return nil;
    }
    const Want want{handler ? handlerTakesReflection : reflection != nullptr, options};
    checkReflectionOptions(self, _cmd, want);
    ComputeDescriptor wire_;
    BuildComputeDescriptor(self, self, _cmd, descriptor, wire_);
    if (!handler) {
        wire::CreateComputePipeline request;
        request.options = options;
        request.reflection = reflection != nullptr;
        request.error = error != nullptr;
        ComputeResult result = createCompute<wire::CreateComputePipeline, wire::CreateComputePipelineReply>(
            self, _cmd, descriptor, request, want, wire_);
        if (error)
            *error = result.error;
        if (reflection)
            *reflection = result.reflection;
        return result.state;
    }
    // The host compiles before it replies; the reply is collected here, so that a failure the plugin reports is
    // raised on the caller's thread, and the handler runs on the completion queue as the driver's does.
    wire::CreateComputePipelineAsync request;
    request.options = options;
    request.completionHandler = true;
    ComputeResult result = createCompute<wire::CreateComputePipelineAsync, wire::CreateComputePipelineAsyncReply>(
        self, _cmd, descriptor, request, want, wire_);
    dispatch_async(completionQueue(), ^{
        handler(result.state, result.reflection, result.error);
    });
    return nil;
}

id<MTLRenderPipelineState> newRenderPipeline(WebMetalDevice *self, SEL _cmd, MTLRenderPipelineDescriptor *descriptor,
                                             MTLPipelineOption options, bool optionsForm,
                                             MTLAutoreleasedRenderPipelineReflection *reflection, NSError **error,
                                             MTLNewRenderPipelineStateWithReflectionCompletionHandler handler,
                                             bool handlerTakesReflection)
{
    RequireShaderEntryPoints(self, _cmd);
    options = compileOptions(self, _cmd, options);
    if (error)
        *error = nil;
    if (NSError *invalid = validate(self, descriptor)) {
        if (handler) {
            handler(nil, nil, invalid);
            return nil;
        }
        if (error)
            *error = invalid;
        return nil;
    }
    if (handler)
        descriptor = [descriptor copy];
    // The driver serializes the vertex descriptor against the vertex function with Metal.framework's
    // newSerializedVertexDataWithFlags:options:error:; its failure (a stage-input function without a matching vertex
    // descriptor) is the pipeline's error. The serialized data is the driver's compile input; the host receives the
    // descriptor itself.
    NSError *serializeError = nil;
    using Serialize = id (*)(id, SEL, uint64_t, uint64_t, NSError **);
    id serialized = ((Serialize)objc_msgSend)(descriptor, sel_registerName("newSerializedVertexDataWithFlags:options:error:"),
                                              kCompilerFlags, options, &serializeError);
    if (!serialized) {
        if (error)
            *error = serializeError;
        if (handler)
            handler(nil, nil, serializeError);
        return nil;
    }
    for (NSString *stage in @[@"fragment", @"vertex", @"mesh", @"object"]) {
        id<MTLFunction> function = [(id)descriptor valueForKey:[stage stringByAppendingString:@"Function"]];
        if (!function)
            continue;
        NSArray *preloaded = [(id)descriptor valueForKey:[stage stringByAppendingString:@"PreloadedLibraries"]];
        NSError *loadError = nil;
        if (![self loadDynamicLibrariesForFunction:function insertLibraries:preloaded error:&loadError]) {
            if (error)
                *error = loadError;
            if (handler)
                handler(nil, nil, loadError);
            return nil;
        }
    }
    const Want want{handler ? handlerTakesReflection : reflection != nullptr, options};
    checkReflectionOptions(self, _cmd, want);
    RenderDescriptor wire_;
    BuildRenderDescriptor(self, self, _cmd, descriptor, wire_);
    RenderResult result;
    if (!handler && !optionsForm) {
        wire::CreateRenderPipeline request;
        request.error = error != nullptr;
        result = createRender<wire::CreateRenderPipeline, wire::CreateRenderPipelineReply>(self, _cmd, descriptor,
                                                                                            request, want, wire_, false);
    } else if (!handler) {
        wire::CreateRenderPipelineReflection request;
        request.options = options;
        request.reflection = reflection != nullptr;
        request.error = error != nullptr;
        result = createRender<wire::CreateRenderPipelineReflection, wire::CreateRenderPipelineReflectionReply>(
            self, _cmd, descriptor, request, want, wire_, true);
    } else if (!optionsForm) {
        wire::CreateRenderPipelineAsync request;
        request.completionHandler = true;
        result = createRender<wire::CreateRenderPipelineAsync, wire::CreateRenderPipelineAsyncReply>(
            self, _cmd, descriptor, request, want, wire_, false);
    } else {
        wire::CreateRenderPipelineReflectionAsync request;
        request.options = options;
        request.completionHandler = true;
        result = createRender<wire::CreateRenderPipelineReflectionAsync, wire::CreateRenderPipelineReflectionAsyncReply>(
            self, _cmd, descriptor, request, want, wire_, true);
    }
    if (!handler) {
        if (error)
            *error = result.error;
        if (reflection)
            *reflection = result.reflection;
        return result.state;
    }
    dispatch_async(completionQueue(), ^{
        handler(result.state, result.reflection, result.error);
    });
    return nil;
}

}  // namespace

@implementation WebMetalDevice (WebMetalShaders)

#pragma mark - Libraries

// -[AGXA10FamilyDevice newDefaultLibrary] (AGXMetalA10 0x21c8d9018), newLibraryWithFile:error: (0x21c8d9068),
// newLibraryWithSource:options:error: (0x21c8d90b8), newLibraryWithSource:options:completionHandler: (0x21c8d9108) and
// newLibraryWithData:error: (0x21c8d9158) each set a bit of the driver's usage telemetry, which has no observable
// effect, and call _MTLDevice's method: Metal.framework parses metallibs itself and compiles source through
// MTLCompilerService (MTLLibraryBuilder::newLibraryWithSource, Metal 0x187ab1e3c), which returns the AIR library the
// framework then builds the library from. The driver takes no part.
- (id<MTLLibrary>)newDefaultLibrary
{
    return [super newDefaultLibrary];
}

- (id<MTLLibrary>)newLibraryWithFile:(NSString *)path error:(NSError **)error
{
    return [super newLibraryWithFile:path error:error];
}

- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source options:(MTLCompileOptions *)options error:(NSError **)error
{
    return [super newLibraryWithSource:source options:options error:error];
}

- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler
{
    [super newLibraryWithSource:source options:options completionHandler:completionHandler];
}

- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data error:(NSError **)error
{
    return [super newLibraryWithData:data error:error];
}

// -[AGXA10FamilyDevice compiler] (AGXMetalA10 0x21c8d5020) answers the Metal.framework MTLCompiler that setupCompiler:
// (0x21c8d92cc) binds to the AGX back end (AGXCompilerCore) with compiler flags 0x21c. The plugin has no back end: the
// host compiles AIR, so the device has no MTLCompiler, which is also _MTLDevice's own answer (Metal 0x187b5770c).
// Metal.framework's function specialization reads only the flags' 0x80 bit from it, which the A10X's compiler clears
// as well (-[_MTLFunctionInternal newSpecializedFunctionWithDescriptor:destinationArchive:functionCache:sync:
// completionHandler:], Metal 0x187aac790-0x187aac798); the specialization itself is MTLCompilerService's.
- (id)compiler
{
    return nil;
}

#pragma mark - Compute pipelines

// -[AGXA10FamilyDevice newComputePipelineStateWithDescriptor:error:] (AGXMetalA10 0x21c8d89cc).
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                                               error:(NSError **)error
{
    return newComputePipeline(self, _cmd, descriptor, MTLPipelineOptionNone, nullptr, error, nil, false);
}

// -[AGXA10FamilyDevice newComputePipelineStateWithDescriptor:options:reflection:error:] (AGXMetalA10 0x21c8d8a9c).
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                                             options:(MTLPipelineOption)options
                                                          reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                                               error:(NSError **)error
{
    return newComputePipeline(self, _cmd, descriptor, options, reflection, error, nil, false);
}

// -[AGXA10FamilyDevice newComputePipelineStateWithDescriptor:completionHandler:] (AGXMetalA10 0x21c8d8a28): the
// handler receives no reflection.
- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                            completionHandler:(MTLNewComputePipelineStateCompletionHandler)completionHandler
{
    MTLNewComputePipelineStateWithReflectionCompletionHandler handler =
        ^(id<MTLComputePipelineState> state, MTLComputePipelineReflection *, NSError *error) {
            completionHandler(state, error);
        };
    newComputePipeline(self, _cmd, descriptor, MTLPipelineOptionNone, nullptr, nullptr, handler, false);
}

// -[AGXA10FamilyDevice newComputePipelineStateWithDescriptor:options:completionHandler:] (AGXMetalA10 0x21c8d8aec).
- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:(MTLNewComputePipelineStateWithReflectionCompletionHandler)completionHandler
{
    newComputePipeline(self, _cmd, descriptor, options, nullptr, nullptr, completionHandler, true);
}

#pragma mark - Render pipelines

// -[AGXA10FamilyDevice newRenderPipelineStateWithDescriptor:error:] (AGXMetalA10 0x21c8d883c).
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                                             error:(NSError **)error
{
    return newRenderPipeline(self, _cmd, descriptor, MTLPipelineOptionNone, false, nullptr, error, nil, false);
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithDescriptor:options:reflection:error:] (AGXMetalA10 0x21c8d890c).
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                                           options:(MTLPipelineOption)options
                                                        reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                                             error:(NSError **)error
{
    return newRenderPipeline(self, _cmd, descriptor, options, true, reflection, error, nil, false);
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithDescriptor:completionHandler:] (AGXMetalA10 0x21c8d8898): the
// handler receives no reflection.
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                           completionHandler:(MTLNewRenderPipelineStateCompletionHandler)completionHandler
{
    MTLNewRenderPipelineStateWithReflectionCompletionHandler handler =
        ^(id<MTLRenderPipelineState> state, MTLRenderPipelineReflection *, NSError *error) {
            completionHandler(state, error);
        };
    newRenderPipeline(self, _cmd, descriptor, MTLPipelineOptionNone, false, nullptr, nullptr, handler, false);
}

// -[AGXA10FamilyDevice newRenderPipelineStateWithDescriptor:options:completionHandler:] (AGXMetalA10 0x21c8d895c).
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                           completionHandler:(MTLNewRenderPipelineStateWithReflectionCompletionHandler)completionHandler
{
    newRenderPipeline(self, _cmd, descriptor, options, true, nullptr, nullptr, completionHandler, true);
}

#pragma mark - Dynamic libraries

// The A10X does not support dynamic libraries (supportsDynamicLibraries NO). -[AGXA10FamilyDevice newDynamicLibrary:
// error:] (AGXMetalA10 0x21c8d8f20) and newDynamicLibrary:computeDescriptor:error: (0x21c8d8f40) let Metal.framework's
// validateDynamicLibrary:state:error: describe the error and return nil; newDynamicLibraryWithURL:error: (0x21c8d8f5c),
// newDynamicLibraryWithURL:options:error: (0x21c8d8f74) and newDynamicLibraryWithDescriptor:error: (0x21c8d8f08) do
// the same with validateDynamicLibraryURL:error: and validateDynamicLibraryDescriptor:error:.
- (id<MTLDynamicLibrary>)newDynamicLibrary:(id<MTLLibrary>)library error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    [self validateDynamicLibrary:library state:NO error:error];
    return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibrary:(id<MTLLibrary>)library computeDescriptor:(id)descriptor error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    [self validateDynamicLibrary:library state:NO error:error];
    return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithURL:(NSURL *)url error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    [self validateDynamicLibraryURL:url error:error];
    return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithURL:(NSURL *)url options:(NSUInteger)options error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    [self validateDynamicLibraryURL:url error:error];
    return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithDescriptor:(id)descriptor error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    [self validateDynamicLibraryDescriptor:descriptor error:error];
    return nil;
}

#pragma mark - Indirect command buffers

// -[AGXA10FamilyDevice newIndirectCommandBufferWithDescriptor:maxCommandCount:options:] (AGXMetalA10 0x21c8d6358)
// creates an AGXA10FamilyIndirectCommandBuffer whose memory holds commands in the A10X's hardware layout, which the
// CPU-side command setters write and the GPU executes. The wire schema defines no WebMetal layout for that memory yet
// (CreateIndirectCommandBuffer, wire/schema/webmetal.toml), so an indirect command buffer cannot be made.
- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                                                       maxCommandCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
    WMLoudMethod(self, _cmd, "indirect command buffers need a command layout the wire schema does not define yet");
}

- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                                                              maxCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
    WMLoudMethod(self, _cmd, "indirect command buffers need a command layout the wire schema does not define yet");
}

@end
