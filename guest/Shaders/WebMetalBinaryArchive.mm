// Binary archives (AGXA10FamilyBinaryArchive on 21H461).
//
// The driver's archive is Metal.framework's _MTLBinaryArchive, which loads, keeps and serializes the archive's
// contents; the driver adds only the pipeline methods that compile functions into it. The plugin's archive is the same
// framework object, and the host keeps the records of what the plugin added (BinaryArchiveAdd*PipelineFunctions), which
// pipelines created with MTLPipelineOptionFailOnBinaryArchiveMiss are looked up in.

#import "WebMetalPipelineDescriptors.h"
#import "WebMetalShaders.h"
#import "../Commands/WebMetalCommands.h"

#import <objc/message.h>

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

// The archive SPI protocol the driver class adopts; Metal.framework defines it.
@protocol MTLBinaryArchiveSPI <MTLBinaryArchive>
@end

// Metal.framework's methods of the base and descriptor (checked with the shader path's entry points,
// CheckShaderEntryPoints).
@interface _MTLBinaryArchive (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(MTLBinaryArchiveDescriptor *)descriptor
                         error:(NSError **)error;
- (BOOL)storeComputePipelineDescriptor:(MTLComputePipelineDescriptor *)descriptor;
- (BOOL)storeRenderPipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor;
@end

@interface MTLBinaryArchiveDescriptor (WebMetalFramework)
- (NSUInteger)options;
- (NSUInteger)writingOptions;
@end

@interface WebMetalBinaryArchive : _MTLBinaryArchive <MTLBinaryArchiveSPI>
@end

@interface WebMetalBinaryArchive () {
@public
    webmetal::Connection *_connection;
    uint64_t _handle;
}
@end

namespace {

using namespace webmetal;

// The front of the driver's archive methods (-[AGXA10FamilyBinaryArchive addComputePipelineFunctionsWithDescriptor:
// options:error:], AGXMetalA10 0x21c8a5f94, and the render form at 0x21c8a59fc), as in pipeline creation: a nil or
// invalid descriptor fails with the driver's error, code 2.
NSError *validate(id device, id descriptor)
{
    if (!descriptor)
        return DriverError(2, @"pipeline descriptor is nil");
    NSError *reason = nil;
    using Validate = BOOL (*)(id, SEL, id, NSError **);
    if (((Validate)objc_msgSend)(descriptor, sel_registerName("validateWithDevice:error:"), device, &reason))
        return nil;
    return DriverError(2, reason.localizedDescription);
}

}  // namespace

@implementation WebMetalBinaryArchive

- (void)dealloc
{
    if (_handle) {
        ForgetHandle((__bridge const void *)self);
        ReleaseHandle(_connection, _handle);
    }
}

// -[AGXA10FamilyBinaryArchive addComputePipelineFunctionsWithDescriptor:options:error:] (AGXMetalA10 0x21c8a5f94):
// the descriptor is stored in the archive's pipeline collection (-storeComputePipelineDescriptor:), checked as pipeline
// creation checks it, and its functions compiled into the archive. The host records the pipeline; the record carries
// no options, so options other than none cannot be served.
- (BOOL)addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                          options:(NSUInteger)options
                                            error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    WebMetalDevice *device = (WebMetalDevice *)[self device];
    [self storeComputePipelineDescriptor:descriptor];
    if (error)
        *error = nil;
    if (NSError *invalid = validate(device, descriptor)) {
        if (error)
            *error = invalid;
        return NO;
    }
    if (id<MTLFunction> function = descriptor.computeFunction) {
        if (![device loadDynamicLibrariesForFunction:function insertLibraries:descriptor.preloadedLibraries
                                               error:error])
            return NO;
    }
    if (options)
        WMLoudArgument(self, _cmd, options, "BinaryArchiveAddComputePipelineFunctions carries no options");
    ComputeDescriptor wire_;
    BuildComputeDescriptor(device, self, _cmd, descriptor, wire_);
    wire::BinaryArchiveAddComputePipelineFunctions request;
    request.archive = _handle;
    request.descriptor = wire_.descriptor;
    request.error = error != nullptr;
    wire::Reply<wire::BinaryArchiveAddComputePipelineFunctionsReply> reply;
    _connection->Call(self, _cmd, request, &reply);
    if (!reply.ok) {
        if (error)
            *error = MetalError(reply.failure);
        return NO;
    }
    return YES;
}

// -[AGXA10FamilyBinaryArchive addComputePipelineFunctionsWithDescriptor:error:] (AGXMetalA10 0x21c8a62c0): the options
// form with none.
- (BOOL)addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor *)descriptor error:(NSError **)error
{
    return [self addComputePipelineFunctionsWithDescriptor:descriptor options:0 error:error];
}

// -[AGXA10FamilyBinaryArchive addRenderPipelineFunctionsWithDescriptor:options:error:] (AGXMetalA10 0x21c8a59fc).
- (BOOL)addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                         options:(NSUInteger)options
                                           error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    WebMetalDevice *device = (WebMetalDevice *)[self device];
    [self storeRenderPipelineDescriptor:descriptor];
    if (error)
        *error = nil;
    if (NSError *invalid = validate(device, descriptor)) {
        if (error)
            *error = invalid;
        return NO;
    }
    if (options)
        WMLoudArgument(self, _cmd, options, "BinaryArchiveAddRenderPipelineFunctions carries no options");
    RenderDescriptor wire_;
    BuildRenderDescriptor(device, self, _cmd, descriptor, wire_);
    wire::BinaryArchiveAddRenderPipelineFunctions request;
    request.archive = _handle;
    request.descriptor = wire_.descriptor;
    request.error = error != nullptr;
    wire::Reply<wire::BinaryArchiveAddRenderPipelineFunctionsReply> reply;
    _connection->Call(self, _cmd, request, &reply);
    if (!reply.ok) {
        if (error)
            *error = MetalError(reply.failure);
        return NO;
    }
    return YES;
}

// -[AGXA10FamilyBinaryArchive addRenderPipelineFunctionsWithDescriptor:error:] (AGXMetalA10 0x21c8a5f48).
- (BOOL)addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor error:(NSError **)error
{
    return [self addRenderPipelineFunctionsWithDescriptor:descriptor options:0 error:error];
}

@end

@implementation WebMetalDevice (WebMetalBinaryArchives)

// -[AGXA10FamilyDevice newBinaryArchiveWithDescriptor:error:] (AGXMetalA10 0x21c8d8f90) is
// -[AGXA10FamilyBinaryArchive initWithDevice:descriptor:error:], which is _MTLBinaryArchive's: Metal.framework opens
// and checks the descriptor's file, if any, and its failures are the call's. The host then holds the archive's records.
- (id<MTLBinaryArchive>)newBinaryArchiveWithDescriptor:(MTLBinaryArchiveDescriptor *)descriptor error:(NSError **)error
{
    RequireShaderEntryPoints(self, _cmd);
    WebMetalBinaryArchive *archive = [[WebMetalBinaryArchive alloc] initWithDevice:self descriptor:descriptor error:error];
    if (!archive)
        return nil;
    Connection *connection = DeviceConnection(self);
    wire::CreateBinaryArchive request;
    request.archive = connection->NewHandle();
    request.descriptor.options = descriptor.options;
    request.descriptor.writingOptions = [(id)descriptor writingOptions];
    NSString *url = descriptor.url.absoluteString;
    std::string text = url.UTF8String ?: "";
    if (url)
        request.descriptor.url = wire::str(text);
    request.error = error != nullptr;
    wire::Reply<wire::CreateBinaryArchiveReply> reply;
    connection->Call(self, _cmd, request, &reply);
    if (!reply.ok) {
        if (error)
            *error = MetalError(reply.failure);
        return nil;
    }
    archive->_connection = connection;
    archive->_handle = request.archive;
    RegisterHandle(archive, request.archive);
    return archive;
}

@end
