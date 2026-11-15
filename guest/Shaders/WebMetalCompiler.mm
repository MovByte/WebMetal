// The device's compiler object (-[AGXA10FamilyDevice compiler]).
//
// On 21H461 the A10X driver answers the Metal.framework MTLCompiler that -[AGXA10FamilyDevice setupCompiler:]
// (AGXMetalA10 0x21c8d92cc) binds to the AGX back end (AGXCompilerCore, in MTLCompilerService) with compiler flags 0x21c
// (the call at 0x21c8d9e94). Metal.framework sends it the back-end requests: pipeline and visible-function compiles,
// function reflection (-[_MTLFunction reflectionWithOptions:binaryArchives:], Metal 0x187a9fbbc, which
// newArgumentEncoderWithBufferIndex: uses), binary-archive and dynamic-library serialization, and the shader-cache
// queries of _MTLDevice. Function specialization reads only its flags (-[_MTLFunctionInternal
// newSpecializedFunctionWithDescriptor:destinationArchive:functionCache:sync:completionHandler:], Metal 0x187aac790:
// compilerFlags & 0x80) and runs in MTLCompilerService itself.
//
// WebMetal has no back end in the guest: the host compiles AIR. The device's compiler object answers the flags the
// A10X's compiler has, so that specialization requests are the A10X's, and function reflection, whose reply the host
// writes as MTLCompilerService writes it (ReflectFunction). Every other MTLCompiler method of 21H461 is a LOUD stub, so
// that a framework path that needs the back end reports instead of receiving nothing.

#import "WebMetalShaders.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <mach-o/dyld.h>

#include <cstring>

@interface WebMetalCompiler : NSObject {
@public
    __weak AGXA10FamilyDevice *_device;
}
- (uint64_t)compilerFlags;
@end

// The completion handler of MTLCompiler's reflection requests: the compiler's status (0 when the request succeeded), the
// reply's reflection data and the compiler's message (the block signature v32@?0Q8@"NSObject<OS_dispatch_data>"16@"NSString"24
// of -[_MTLFunction reflectionWithOptions:binaryArchives:]'s handler, Metal 0x187c0e3fa).
typedef void (^WebMetalReflectionHandler)(NSUInteger status, dispatch_data_t reflection, NSString *message);

// Metal.framework's MTLTargetDeviceArchitecture setters (21H461: setCpuType: v20@0:8I16, setSubType: v20@0:8I16,
// setDriverVersion: v24@0:8Q16, setVersion: v20@0:8{MTLTargetDeviceArchVersion=SCC}16).
struct WebMetalTargetDeviceArchVersion {
    uint16_t major;
    uint8_t minor;
    uint8_t patch;
};

@protocol WebMetalTargetDeviceArchitecture
- (void)setCpuType:(uint32_t)type;
- (void)setSubType:(uint32_t)type;
- (void)setDriverVersion:(uint64_t)version;
- (void)setVersion:(WebMetalTargetDeviceArchVersion)version;
@end

namespace {

// The instance methods of Metal.framework's MTLCompiler on 21H461 (its method list in the dyld shared cache), with their
// type encodings, other than its initializer, its C++ constructor and destructor, -dealloc, -compilerFlags and the
// reflection requests WebMetalCompiler implements.
struct CompilerSelector {
    const char *name;
    const char *types;
};

const CompilerSelector kCompilerSelectors[] = {
    {"addComputePipelineStateWithDescriptor:destinationBinaryArchive:options:error:", "B48@0:8@16@24Q32^@40"},
    {"addFunctionKeys:function:driverData:frameworkData:compilerOptions:", "v52@0:8^{RequiredFunctionKeys=[8@][8@]}16@24@32@40i48"},
    {"addLegacyCompiledOutput:cachedCompiledOutput:importedSymbols:importedLibraries:hashKey:type:functionCache:", "v76@0:8@16@24@32@40@48c56{shared_ptr<MultiLevelPipelineCache>=^{MultiLevelPipelineCache}^{__shared_weak_count}}60"},
    {"addLegacyCompiledOutput:importedSymbols:importedLibraries:hashKey:type:functionCache:", "v68@0:8@16@24@32@40c48{shared_ptr<MultiLevelPipelineCache>=^{MultiLevelPipelineCache}^{__shared_weak_count}}52"},
    {"addMeshRenderPipelineStateWithDescriptor:destinationBinaryArchive:options:error:", "B48@0:8@16@24Q32^@40"},
    {"addRenderPipelineStateWithDescriptor:destinationBinaryArchive:options:error:", "B48@0:8@16@24Q32^@40"},
    {"addTileRenderPipelineStateWithDescriptor:destinationBinaryArchive:options:error:", "B48@0:8@16@24Q32^@40"},
    {"allowLibrariesFromOtherPlatforms", "v16@0:8"},
    {"cacheLegacyCompilerData:cachedData:hashMap:libraryData:functionCache:", "v64@0:8^v16^v24^v32^{MTLLibraryData=^^?{?=[32C]}Ai@@}40{shared_ptr<MultiLevelPipelineCache>=^{MultiLevelPipelineCache}^{__shared_weak_count}}48"},
    {"compileDynamicLibraryWithDescriptor:computePipelineDescriptor:completionHandler:", "v40@0:8@16@24@?32"},
    {"compileDynamicLibraryWithDescriptor:computePipelineDescriptor:error:", "@40@0:8@16@24^@32"},
    {"compileFunction:frameworkData:driverKeyData:options:pipelineCache:completionHandler:", "v64@0:8@16@24@32Q40@48@?56"},
    {"compileFunction:frameworkData:driverKeyData:options:pipelineCache:sync:completionHandler:", "v68@0:8@16@24@32Q40@48B56@?60"},
    {"compileFunction:serializedData:stateData:options:completionHandler:", "v56@0:8@16@24@32Q40@?48"},
    {"compileFunction:serializedPipelineData:stateData:linkDataSize:frameworkLinking:options:pipelineCache:sync:completionHandler:", "v76@0:8@16@24@32Q40B48I52@56B64@?68"},
    {"compileFunction:serializedPipelineData:stateData:linkDataSize:frameworkLinking:options:sync:completionHandler:", "v68@0:8@16@24@32Q40B48I52B56@?60"},
    {"compileFunction:visibleFunctions:privateVisibleFunctions:visibleFunctionGroups:frameworkData:driverKeyData:options:pipelineCache:sync:completionHandler:", "v92@0:8@16@24@32@40@48@56Q64@72B80@?84"},
    {"compileFunction:visibleFunctions:visibleFunctionGroups:frameworkData:driverKeyData:options:pipelineCache:sync:completionHandler:", "v84@0:8@16@24@32@40@48Q56@64B72@?76"},
    {"compileFunctionRequest:completionHandler:", "v32@0:8@16@?24"},
    {"compileFunctionRequestInternal:frameworkLinking:linkDataSize:reflectionOnly:completionHandler:", "v48@0:8@16B24Q28B36@?40"},
    {"compileLibraryRequest:completionHandler:", "v64@0:8{MTLCompileLibraryRequestData=B@I@@}16@?56"},
    {"compileRequest:binaryArchives:sync:completionHandler:", "v44@0:8@16@24B32@?36"},
    {"compileRequest:completionHandler:", "v32@0:8@16@?24"},
    {"compileRequest:pipelineCache:completionHandler:", "v40@0:8@16@24@?32"},
    {"compileRequest:pipelineCache:sync:completionHandler:", "v44@0:8@16@24B32@?36"},
    {"compileRequestInternal:binaryArchives:pipelineCache:sync:completionHandler:", "v52@0:8@16@24@32B40@?44"},
    {"compileStatelessFunctionRequest:reflectionOnly:completionHandler:", "v36@0:8@16B24@?28"},
    {"compilerConnectionManager", "^{MTLCompilerConnectionManager=^^?Bi}16@0:8"},
    {"compilerId", "I16@0:8"},
    {"compilerQueue", "@16@0:8"},
    {"computeVariantEntryWithDescriptor:airDescriptor:options:serializedComputeDataDescriptor:asyncCompile:pipelineCache:destinationBinaryArchive:computeProgram:kernelDriverCompileTimeData:compileTimeStatistics:", "^{VariantEntry=*QQQ@@Q@@}92@0:8@16@24Q32@40B48@52@60^^{MTLProgramObject}68^@76@84"},
    {"copyShaderCacheToPath:", "B24@0:8@16"},
    {"createBinaryArchiveWithCompletionHanlder:", "v24@0:8@?16"},
    {"createMeshStageAndLinkPipelineWithFragment:fragmentVariant:objectFunction:serializedObjectDescriptor:meshFunction:serializedMeshDescriptor:descriptor:airDescriptor:destinationArchive:options:reflection:compileStatistics:fragmentCompileTimeData:pipelineArchiverId:error:completionHandler:", "@144@0:8^v16@24@32@40@48@56@64@72@80Q88^@96@104@112@120^@128@?136"},
    {"createVertexStageAndLinkPipelineWithFragment:fragmentVariant:vertexFunction:serializedVertexDescriptor:descriptor:airDescriptor:destinationArchive:options:reflection:compileStatistics:fragmentCompileTimeData:pipelineArchiverId:error:completionHandler:", "@128@0:8^v16@24@32@40@48@56@64Q72^@80@88@96@104^@112@?120"},
    {"generateMachOWithID:binaryEntries:numEntries:machOSpecializedData:machOType:Path:platform:completionHandler:", "v76@0:8r*16^{machOEntry=Q^{MTLFunctionId}@@@@}24Q32r^v40i48@52Q60@?68"},
    {"getDataForScript:device:function:functionType:compilerOptions:compilerOptionsSize:compiledNextStageVariant:dataSize:", "^v80@0:8r^v16@24@32Q40^*48Q56@64^Q72"},
    {"getFunctionId:", "{MTLFunctionId={?=[32C]}[4{?=[32C]}]I}24@0:8@16"},
    {"getFunctionId:airScript:vendorPluginFunctionId:", "{MTLFunctionId={?=[32C]}[4{?=[32C]}]I}40@0:8@16r^v24^{?=[32C]}32"},
    {"getFunctionKeyWithComputePipelineState:options:unfilteredOptions:airScript:keySize:", "^v56@0:8@16Q24Q32r^v40^Q48"},
    {"getFunctionKeyWithRenderPipelineDescriptor:options:unfilteredOptions:airScript:function:functionType:compiledNextStageVariant:keySize:", "^v80@0:8@16Q24Q32r^v40@48Q56@64^Q72"},
    {"getFunctionKeyWithRenderPipelineDescriptor:options:unfilteredOptions:airScript:function:functionType:compiledNextStageVariant:keySize:functionDriverData:functionDriverSize:", "^v96@0:8@16Q24Q32r^v40@48Q56@64^Q72^^v80^Q88"},
    {"getFunctionKeyWithTileRenderPipelineDescriptor:options:unfilteredOptions:function:functionType:airScript:keySize:", "^v72@0:8@16Q24Q32@40Q48r^v56^Q64"},
    {"getGPUCompilerHashForScript:type:", "{?=[32C]}32@0:8r^v16Q24"},
    {"getHashForScript:device:function:functionType:compilerOptions:compilerOptionsSize:compiledNextStageVariant:", "^v72@0:8r^v16@24@32Q40^*48Q56@64"},
    {"getProgramObject:destinationArchive:sourceBinaryArchives:variantKey:requiredKeys:failOnMiss:", "^v60@0:8@16@24@32r^{VariantKey=^v^vQQ@^vQ}40r^{RequiredFunctionKeys=[8@][8@]}48B56"},
    {"getProgramObjectForFunction:variantKey:requiredKeys:sourceBinaryArchives:", "^v48@0:8@16r^{VariantKey=^v^vQQ@^vQ}24r^{RequiredFunctionKeys=[8@][8@]}32@40"},
    {"getShaderCacheKeys", "^v16@0:8"},
    {"hashKeyForLibraryRequest:", "{?=[32C]}24@0:8^{MTLCompileLibraryRequestData=B@I@@}16"},
    {"hashKeyForStatelessCompilationRequest:", "{?=[32C]}24@0:8@16"},
    {"initializeFunctionRequestScriptAndFunctionId:script:driverCompilerOptions:compiledNextStageVariant:", "v48@0:8@16r^v24@32@40"},
    {"initializeVendorPluginFunctionId:airScript:driverCompilerOptions:compiledNextStageVariant:vendorPlugin:", "B56@0:8@16r^v24@32@40^{?=[32C]}48"},
    {"libraryCacheStats", "{?=II}16@0:8"},
    {"newComputePipelineStateWithDescriptor:options:reflection:error:completionHandler:", "@56@0:8@16Q24^@32^@40@?48"},
    {"newComputePipelineStateWithDescriptorInternal:options:pipelineCache:destinationBinaryArchive:reflection:error:completionHandler:", "@72@0:8@16Q24@32@40^@48^@56@?64"},
    {"newRenderPipelineStateWithDescriptor:options:reflection:error:completionHandler:", "@56@0:8@16Q24^@32^@40@?48"},
    {"newRenderPipelineStateWithDescriptorInternal:options:reflection:destinationBinaryArchive:error:completionHandler:", "@64@0:8@16Q24^@32@40^@48@?56"},
    {"newRenderPipelineStateWithTileDescriptor:options:reflection:error:completionHandler:", "@56@0:8@16Q24^@32^@40@?48"},
    {"newRenderPipelineStateWithTileDescriptorInternal:options:reflection:destinationBinaryArchive:error:completionHandler:", "@64@0:8@16Q24^@32@40^@48@?56"},
    {"pipelineCacheStats", "{?=II}16@0:8"},
    {"pipelineStateWithVariant:descriptor:options:computeProgram:kernelDriverCompileTimeData:serializedComputeDataDescriptor:compileTimeStatistics:reflection:error:completionHandler:", "@96@0:8^{VariantEntry=*QQQ@@Q@@}16@24Q32^v40@48@56@64^@72^@80@?88"},
    {"pluginPath", "@16@0:8"},
    {"releaseCompilerOutputBlocks:", "v24@0:8^v16"},
    {"releaseCompilerOutputBlocks:hashMap:", "v32@0:8^v16^v24"},
    {"releaseHashMaps:", "v24@0:8^v16"},
    {"renderPipelineStateWithTileVariant:descriptor:options:tileProgram:kernelDriverCompileTimeData:serializedTileDataDescriptor:compileTimeStatistics:reflection:error:completionHandler:", "@96@0:8^{VariantEntry=*QQQ@@Q@@}16@24Q32^v40@48@56@64^@72^@80@?88"},
    {"requiredKeysForFunction:variantKey:frameworkData:compilerOptions:driverCompilerOptions:airScript:archives:compiledNextStageVariant:", "{RequiredFunctionKeys=[8@][8@]}76@0:8@16r^{VariantKey=^v^vQQ@^vQ}24@32i40@44r^v52@60@68"},
    {"statelessBackendCompileRequestInternal:sync:compilerHash:reflectionOnly:completionHandler:", "v48@0:8^{MTLCompilerFunctionRequest=^^?i@I@@III@@Q@@@@iBIB*^v@Q@@}16B24r^{?=[32C]}28B36@?40"},
    {"tileVariantEntryWithDescriptor:airDescriptor:options:serializedTileDataDescriptor:asyncCompile:destinationBinaryArchive:tileProgram:kernelDriverCompileTimeData:compileTimeStatistics:", "^{VariantEntry=*QQQ@@Q@@}84@0:8@16@24Q32@40B48@52^^{MTLProgramObject}60^@68@76"},
    {"unloadShaderCaches", "v16@0:8"},
    {"validateLanguageAndAIRVersionForFunction:completionHandler:", "B32@0:8@16@?24"},
};

void installStubs()
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        Class cls = [WebMetalCompiler class];
        for (const CompilerSelector &s : kCompilerSelectors) {
            if (!class_addMethod(cls, sel_registerName(s.name), (IMP)WMLoudStub, s.types))
                WMReport("WebMetalCompiler", (std::string("refused: a stub for -") + s.name + " cannot be added").c_str());
        }
    });
}

}  // namespace

namespace {

// -[_MTLFunctionInternal functionData] on 21H461: the function's MTLFunctionData (its AIR version at +0x28 and +0x2a,
// its language version at +0x2c and +0x2e).
constexpr char kFunctionDataTypes[] = "r^{MTLFunctionData=QQQQQSSSS{?=[32C]}C@Cb2b6@{?=[32C]}@}16@0:8";

// -[MTLCompiler validateLanguageAndAIRVersionForFunction:completionHandler:] (Metal 0x187a9b72c), which the
// reflection request runs first (compileFunctionRequestInternal:..., 0x187a97f24): nil when the function's versions are
// ones 21H461 compiles, else the message the compiler's handler receives with status 2. Language versions 1.0-1.2,
// 2.0-2.6, 3.0 and 3.1 are accepted (the table at 0x187bfb606 accepts every 2.x up to 2.6); AIR versions 1.6, 1.8-1.11
// and 2.0-2.6. A bad AIR version is reported as such when the language version is accepted.
NSString *versionProblem(id self, SEL _cmd, id<MTLFunction> function)
{
    SEL selector = sel_registerName("functionData");
    Method method = class_getInstanceMethod(object_getClass(function), selector);
    if (!method || strcmp(method_getTypeEncoding(method), kFunctionDataTypes) != 0)
        WMLoudMethod(self, _cmd, "the function has no 21H461 -functionData, whose versions the compiler checks");
    using FunctionData = const uint8_t *(*)(id, SEL);
    const uint8_t *data = ((FunctionData)objc_msgSend)(function, selector);
    auto half = [data](size_t at) {
        uint16_t v;
        memcpy(&v, data + at, sizeof v);
        return static_cast<uint32_t>(v);
    };
    const uint32_t air = half(0x28) << 16 | half(0x2a);
    const uint32_t language = half(0x2c) << 16 | half(0x2e);
    const bool languageOk = (language >= 0x20000 && language - 0x20000 <= 6) || language == 0x30000 ||
                            language == 0x30001 || (language < 0x20000 && language - 0x10000 < 3);
    const bool airOk = air - 0x20000 < 7 || (air - 0x10006 <= 5 && air - 0x10006 != 1);
    if (languageOk && airOk)
        return nil;
    if (languageOk)
        return [NSString stringWithFormat:@"Function %@ has a deployment target (0x%08X) which is incompatible with "
                                          @"this OS (0x%08X).",
                                          function.name, air, 0x20006u];
    return [NSString stringWithFormat:@"Function %@ is using language version %u.%u which is incompatible with this OS.",
                                      function.name, half(0x2c), half(0x2e)];
}

}  // namespace

namespace webmetal {

namespace {

const char kDebugInfoKey = 0;

}  // namespace

bool ShaderDebugInfoCaching(AGXA10FamilyDevice *device)
{
    return [objc_getAssociatedObject(device, &kDebugInfoKey) boolValue];
}

void SetShaderDebugInfoCaching(AGXA10FamilyDevice *device, bool caching)
{
    objc_setAssociatedObject(device, &kDebugInfoKey, @(caching), OBJC_ASSOCIATION_RETAIN);
}

}  // namespace webmetal

@implementation WebMetalCompiler

// -[MTLCompiler compilerFlags] (Metal 0x187aad298) answers the flags it was created with: 0x21c for the A10X's.
- (uint64_t)compilerFlags
{
    return 0x21c;
}

// -[MTLCompiler reflectionWithFunction:options:completionHandler:] (Metal 0x187b90194): not synchronous.
- (void)reflectionWithFunction:(id<MTLFunction>)function
                       options:(uint64_t)options
             completionHandler:(WebMetalReflectionHandler)handler
{
    [self reflectionWithFunction:function options:options sync:NO completionHandler:handler];
}

// -[MTLCompiler reflectionWithFunction:options:sync:completionHandler:] (Metal 0x187ac881c): no pipeline library.
- (void)reflectionWithFunction:(id<MTLFunction>)function
                       options:(uint64_t)options
                          sync:(BOOL)sync
             completionHandler:(WebMetalReflectionHandler)handler
{
    [self reflectionWithFunction:function options:options sync:sync pipelineLibrary:nil completionHandler:handler];
}

// -[MTLCompiler reflectionWithFunction:options:sync:pipelineLibrary:completionHandler:] (Metal 0x187ac8810): no binary
// archives.
- (void)reflectionWithFunction:(id<MTLFunction>)function
                       options:(uint64_t)options
                          sync:(BOOL)sync
               pipelineLibrary:(id)pipelineLibrary
             completionHandler:(WebMetalReflectionHandler)handler
{
    [self reflectionWithFunction:function
                         options:options
                            sync:sync
                 pipelineLibrary:pipelineLibrary
                  binaryArchives:nil
               completionHandler:handler];
}

// -[MTLCompiler reflectionWithFunction:options:sync:binaryArchives:completionHandler:] (Metal 0x187a9ff5c): no pipeline
// library.
- (void)reflectionWithFunction:(id<MTLFunction>)function
                       options:(uint64_t)options
                          sync:(BOOL)sync
                binaryArchives:(NSArray *)binaryArchives
             completionHandler:(WebMetalReflectionHandler)handler
{
    [self reflectionWithFunction:function
                         options:options
                            sync:sync
                 pipelineLibrary:nil
                  binaryArchives:binaryArchives
               completionHandler:handler];
}

// -[MTLCompiler reflectionWithFunction:options:sync:pipelineLibrary:binaryArchives:completionHandler:] (Metal
// 0x187a9fcb0) sends a reflection-only request for the function to MTLCompilerService, after looking for the reply in
// the binary archives and the pipeline library's cache. Its request's pipeline options are (options & 0x20200000) |
// 0x10003, which never include MTLPipelineOptionFailOnBinaryArchiveMiss, so the archives and the cache only decide where
// the reply comes from: what the handler receives is the service's reply, which the host writes (ReflectFunction,
// host/src/pipeline/function_reflection.rs), with status 0 and no message. A synchronous request runs the handler
// before it returns; an asynchronous one runs it on a queue, as the compiler's connection does.
- (void)reflectionWithFunction:(id<MTLFunction>)function
                       options:(uint64_t)options
                          sync:(BOOL)sync
               pipelineLibrary:(id)pipelineLibrary
                binaryArchives:(NSArray *)binaryArchives
             completionHandler:(WebMetalReflectionHandler)handler
{
    // The reply is the legacy reflection of MTLCompiler.framework 343.21, which Metal.framework 343.21 parses
    // (-[MTLFunctionReflectionInternal initWithDevice:reflectionData:functionType:options:], Metal 0x187abe160). Another
    // framework's reader expects another layout (this Mac's, for one, writes version 0x20001 with fields 343.21 has
    // not) and aborts on a layout it does not expect, so the request is refused under any other framework.
    if (NSVersionOfRunTimeLibrary("Metal") != (343 << 16 | 21 << 8))
        WMLoudMethod(self, _cmd, "the running Metal.framework is not 343.21 (21H461), whose reflection reader the host "
                                 "writes for");
    if (!function)
        WMLoudMethod(self, _cmd, "no function: the compiler service's reply to a request without one is not established");
    if (NSString *problem = versionProblem(self, _cmd, function)) {
        handler(2, nil, problem);
        return;
    }
    AGXA10FamilyDevice *device = _device;
    if (!device)
        WMLoudMethod(self, _cmd, "the compiler's device is gone");
    webmetal::Connection *connection = webmetal::DeviceConnection(device);
    webmetal::wire::ReflectFunction request;
    request.function = webmetal::FunctionHandle(device, self, _cmd, function);
    webmetal::wire::Reply<webmetal::wire::ReflectFunctionReply> reply;
    connection->Call(self, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(self, _cmd, ("the host answered a reflection request with a failure Metal defines, which the "
                                  "compiler service's reply does not carry: " + reply.failure.message).c_str());
    dispatch_data_t data = dispatch_data_create(reply.value.reflection.data(), reply.value.reflection.size(), nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (sync) {
        handler(0, data, nil);
        return;
    }
    static dispatch_queue_t queue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        queue = dispatch_queue_create("com.apple.WebMetal.reflection-completion", DISPATCH_QUEUE_SERIAL);
    });
    dispatch_async(queue, ^{
        handler(0, data, nil);
    });
}

@end

@implementation AGXA10FamilyDevice (WebMetalCompiler)

// -[AGXA10FamilyDevice targetDeviceArchitecture] (AGXMetalA10 0x21c8d5088): a new MTLTargetDeviceArchitecture with CPU
// type 0x1000013 and the subtype, driver version and version setupCompiler: recorded for the device's architecture.
// AGXA10XDevice's initializer passes architecture 0x10007 (AGXMetalA10 0x21c9c456c); setupCompiler: maps it to the
// subtype 0x61 (its tables at 0x21c9de4c0 and 0x21c9de560, entry 3) and the compiler name "g9g", and records driver
// version 2 and version 2 (0x21c8d94b8-0x21c8d94c4).
- (id)targetDeviceArchitecture
{
    id<WebMetalTargetDeviceArchitecture> architecture = [objc_getClass("MTLTargetDeviceArchitecture") new];
    if (!architecture)
        WMLoudMethod(self, _cmd, "the running Metal.framework has no MTLTargetDeviceArchitecture");
    [architecture setCpuType:0x1000013];
    [architecture setSubType:0x61];
    [architecture setDriverVersion:2];
    [architecture setVersion:WebMetalTargetDeviceArchVersion{2, 0, 0}];
    return architecture;
}

// -[AGXA10FamilyDevice shaderDebugInfoCaching] and setShaderDebugInfoCaching: (AGXMetalA10 0x21c8d7dac, 0x21c8d7d98)
// read and write a flag of the driver's device, which its initializer clears (0x21c8d32a0). The driver's back-end
// requests carry it (createComputeProgramVariant, 0x21c8a64b4); the host keeps no shader debug information, so pipeline
// creation with it set is LOUD (WebMetalDeviceShaders.mm).
- (BOOL)shaderDebugInfoCaching
{
    return webmetal::ShaderDebugInfoCaching(self);
}

- (void)setShaderDebugInfoCaching:(BOOL)caching
{
    webmetal::SetShaderDebugInfoCaching(self, caching);
}

// -[AGXA10FamilyDevice compiler] (AGXMetalA10 0x21c8d5020): the device's one compiler object.
- (id)compiler
{
    installStubs();
    static const char kKey = 0;
    @synchronized(self) {
        WebMetalCompiler *compiler = objc_getAssociatedObject(self, &kKey);
        if (!compiler) {
            compiler = [WebMetalCompiler new];
            compiler->_device = self;
            objc_setAssociatedObject(self, &kKey, compiler, OBJC_ASSOCIATION_RETAIN);
        }
        return compiler;
    }
}

@end
