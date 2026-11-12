// What the shader path shares (WebMetalShaders.h): the driver's error domain, the framework entry points it calls, the
// registration of functions and their AIR with the host, and the wire handles of pipeline objects.

#import "WebMetalShaders.h"
#import "../Commands/WebMetalCommands.h"

#import <CommonCrypto/CommonDigest.h>
#import <objc/runtime.h>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <set>

namespace webmetal {

NSString *const kDriverErrorDomain = @"AGXMetalA10";

NSError *DriverError(NSInteger code, NSString *message)
{
    NSDictionary *info = [NSDictionary dictionaryWithObject:message forKey:NSLocalizedDescriptionKey];
    return [[NSError alloc] initWithDomain:kDriverErrorDomain code:code userInfo:info];
}

NSError *MetalError(const wire::Failure &failure)
{
    const std::string domain = failure.domain.value_or("");
    NSString *message = [[NSString alloc] initWithBytes:failure.message.data() length:failure.message.size()
                                               encoding:NSUTF8StringEncoding];
    return [NSError errorWithDomain:@(domain.c_str()) code:failure.code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @""}];
}

namespace {

struct EntryPoint {
    const char *cls;
    const char *selector;
    const char *types;
};

// The Metal.framework methods WebMetalShaders.h declares and the shader path sends, with their 21H461 type encodings
// (the class's method list in the 21H461 dyld shared cache).
const EntryPoint kEntryPoints[] = {
    {"_MTLComputePipelineState", "initWithDevice:pipelineStateDescriptor:", "@32@0:8@16@24"},
    {"_MTLRenderPipelineState", "initWithDevice:pipelineStateDescriptor:", "@32@0:8@16@24"},
    {"_MTLBinaryArchive", "initWithDevice:descriptor:error:", "@40@0:8@16@24^@32"},
    {"_MTLBinaryArchive", "storeComputePipelineDescriptor:", "B24@0:8@16"},
    {"_MTLBinaryArchive", "storeRenderPipelineDescriptor:", "B24@0:8@16"},
    {"_MTLIndirectArgumentEncoder", "initWithLayout:device:", "@32@0:8@16@24"},
    {"_MTLIndirectArgumentBufferLayout", "init", "@16@0:8"},
    {"_MTLIndirectArgumentBufferLayout", "setStructType:withDevice:", "v32@0:8@16@24"},
    {"_MTLDevice", "loadDynamicLibrariesForFunction:insertLibraries:error:", "@40@0:8@16@24^@32"},
    {"_MTLDevice", "validateDynamicLibrary:state:error:", "B36@0:8@16B24^@28"},
    {"_MTLDevice", "validateDynamicLibraryURL:error:", "B32@0:8@16^@24"},
    {"_MTLDevice", "validateDynamicLibraryDescriptor:error:", "B32@0:8@16^@24"},
    {"_MTLFunction", "bitcodeData", "@16@0:8"},
    {"MTLComputePipelineDescriptorInternal", "validateWithDevice:error:", "B32@0:8@16^@24"},
    {"MTLRenderPipelineDescriptorInternal", "validateWithDevice:error:", "B32@0:8@16^@24"},
    {"MTLRenderPipelineDescriptorInternal", "newSerializedVertexDataWithFlags:options:error:", "@40@0:8Q16Q24^@32"},
};

// Instance variables of Metal.framework's reflection classes the plugin fills, as the framework's deserializer does
// (-[MTLComputePipelineReflectionInternal initWithSerializedData:serializedStageInputDescriptor:device:options:flags:],
// Metal 0x187abd5ec on 21H461).
const EntryPoint kIvars[] = {
    {"MTLComputePipelineReflectionInternal", "_arguments", "@\"NSArray\""},
    {"MTLComputePipelineReflectionInternal", "_builtInArguments", "@\"NSArray\""},
    {"MTLComputePipelineReflectionInternal", "_bindings", "@\"NSArray\""},
    {"MTLRenderPipelineReflectionInternal", "_vertexArguments", "@\"NSArray\""},
    {"MTLRenderPipelineReflectionInternal", "_fragmentArguments", "@\"NSArray\""},
    {"MTLRenderPipelineReflectionInternal", "_vertexBuiltInArguments", "@\"NSArray\""},
    {"MTLRenderPipelineReflectionInternal", "_vertexBindings", "@\"NSArray\""},
    {"MTLRenderPipelineReflectionInternal", "_fragmentBindings", "@\"NSArray\""},
};

}  // namespace

bool CheckShaderEntryPoints(std::string *problem)
{
    static std::string missing;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        for (const EntryPoint &e : kEntryPoints) {
            Method method = class_getInstanceMethod(objc_getClass(e.cls), sel_registerName(e.selector));
            const char *types = method ? method_getTypeEncoding(method) : nullptr;
            if (!types || strcmp(types, e.types) != 0) {
                missing = std::string("-[") + e.cls + " " + e.selector + "] " +
                          (types ? std::string("has type encoding ") + types : "is missing") + ", expected " + e.types;
                return;
            }
        }
        for (const EntryPoint &e : kIvars) {
            Ivar ivar = class_getInstanceVariable(objc_getClass(e.cls), e.selector);
            const char *types = ivar ? ivar_getTypeEncoding(ivar) : nullptr;
            if (!types || strcmp(types, e.types) != 0) {
                missing = std::string(e.cls) + "." + e.selector + " " +
                          (types ? std::string("has type ") + types : "is missing") + ", expected " + e.types;
                return;
            }
        }
    });
    *problem = missing;
    return missing.empty();
}

void RequireShaderEntryPoints(id self, SEL _cmd)
{
    std::string problem;
    if (!CheckShaderEntryPoints(&problem))
        WMLoudMethod(self, _cmd, ("refused: the running Metal.framework lacks what the shader path uses: " + problem).c_str());
}

namespace {

using Digest = std::array<uint8_t, 32>;

// The AIR modules each connection has registered (RegisterAir is sent once per digest per connection).
std::mutex gAirLock;
std::map<Connection *, std::set<Digest>> gAir;

}  // namespace

}  // namespace webmetal

// The registration of one function object with the host: an associated object of the function, so that the handle is
// released when the function is deallocated.
@interface WebMetalFunctionRegistration : NSObject {
@public
    webmetal::Connection *_connection;
    uint64_t _handle;
}
@end

@implementation WebMetalFunctionRegistration

- (void)dealloc
{
    webmetal::ReleaseHandle(_connection, _handle);
}

@end

namespace webmetal {

namespace {

// The key of a function's registration among its associated objects.
const char kRegistrationKey = 0;
std::mutex gFunctionLock;

}  // namespace

uint64_t FunctionHandle(WebMetalDevice *device, id self, SEL _cmd, id<MTLFunction> function)
{
    if (!function)
        return 0;
    Connection *connection = DeviceConnection(device);
    std::lock_guard<std::mutex> guard(gFunctionLock);
    WebMetalFunctionRegistration *registration = objc_getAssociatedObject(function, &kRegistrationKey);
    if (registration) {
        if (registration->_connection != connection)
            WMLoudMethod(self, _cmd, "the function belongs to another device's connection");
        return registration->_handle;
    }
    // -[_MTLFunction bitcodeData] (Metal 0x187b8b420 on 21H461): the function's AIR module, read from its library at
    // the function's bitcode offset and size; for a specialized function, the module MTLCompilerService returned.
    NSData *bitcode = [(id<WebMetalFunctionBitcode>)function bitcodeData];
    if (!bitcode.length)
        WMLoudMethod(self, _cmd, "the function has no AIR bitcode to hand to the host");
    Digest digest{};
    CC_SHA256(bitcode.bytes, static_cast<CC_LONG>(bitcode.length), digest.data());
    bool known;
    {
        std::lock_guard<std::mutex> air(gAirLock);
        known = gAir[connection].count(digest) != 0;
    }
    if (!known) {
        wire::RegisterAir request;
        request.hash.bytes = digest;
        request.bitcode = {static_cast<const uint8_t *>(bitcode.bytes), bitcode.length};
        wire::Reply<wire::RegisterAirReply> reply;
        connection->Call(self, _cmd, request, &reply);
        if (!reply.ok)
            WMLoudMethod(self, _cmd, ("the host refused the function's AIR: " + reply.failure.message).c_str());
        std::lock_guard<std::mutex> air(gAirLock);
        gAir[connection].insert(digest);
    }
    const std::string name = function.name.UTF8String ?: "";
    wire::RegisterFunction request;
    request.function = connection->NewHandle();
    request.air.bytes = digest;
    request.name = wire::str(name);
    request.functionType = function.functionType;
    // A function Metal.framework specialized carries specialized bitcode (MTLCompilerService's specialization of the
    // module with the constant values); an unspecialized function has no values. No constants travel separately.
    request.constants = {};
    wire::Reply<wire::RegisterFunctionReply> reply;
    connection->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, ("the host refused the function " + name + ": " + reply.failure.message).c_str());
    registration = [WebMetalFunctionRegistration new];
    registration->_connection = connection;
    registration->_handle = request.function;
    objc_setAssociatedObject(function, &kRegistrationKey, registration, OBJC_ASSOCIATION_RETAIN);
    return request.function;
}

std::vector<uint64_t> ObjectHandles(id self, SEL _cmd, NSArray *objects, const char *what)
{
    std::vector<uint64_t> out;
    for (id object in objects) {
        const uint64_t handle = HandleOf(object);
        if (!handle)
            WMLoudMethod(self, _cmd, (std::string("an object in ") + what + " has no WebMetal handle").c_str());
        out.push_back(handle);
    }
    return out;
}

void AdoptPipelineHandle(WebMetalComputePipelineState *pipeline, Connection *connection, uint64_t handle)
{
    pipeline->_connection = connection;
    pipeline->_handle = handle;
    RegisterHandle(pipeline, handle);
}

void AdoptPipelineHandle(WebMetalRenderPipelineState *pipeline, Connection *connection, uint64_t handle)
{
    pipeline->_connection = connection;
    pipeline->_handle = handle;
    RegisterHandle(pipeline, handle);
}

}  // namespace webmetal
