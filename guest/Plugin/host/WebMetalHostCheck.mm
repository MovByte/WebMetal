// Host checks of the guest plugin on this Mac's Metal.framework, through the macOS test shim (WebMetalHostShim.mm).
//
//     WebMetalHostCheck register <shim.dylib> <expectations.json> <maxBufferLength sentinel>
//     WebMetalHostCheck strict   <shim.dylib>    (run with WEBMETAL_STRICT=1: must abort at the first LOUD stub)
//     WebMetalHostCheck refuse   <shim.dylib>    (a shim built with 21H461's layouts: must refuse the device)
//
// register: the device registers through MTLAddDevice and is listed; Metal.framework's limits, computed by its own
// -initLimits from the plugin's answers, equal the device profile's, and the limit the kernel supplies carries the
// sentinel; every capability the profile records is answered as recorded (Metal.framework's own answers where the
// profile's second method found this Mac's framework to agree); every driver selector of the A10X is answered, by the
// plugin or a LOUD stub, and nothing the driver inherits or lacks is overridden; LOUD stubs report and fail; the
// guard accepts this Mac's layouts and refuses 21H461's and altered ones. expectations.json is written from the
// device profile by UniXNU's scripts/build-web-metal-plugin.py. Prints one line per failure and PASS or FAIL.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include "../../Generated/WebMetalTargetSurface.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace webmetal;

namespace {

struct Shim {
    id<MTLDevice> (*registerDevice)(uint64_t);
    int (*checkLayouts)(const LayoutSet *, char *, size_t);
    const LayoutSet *(*expected)(void);
    size_t (*reportCount)(void);
    void (*lastReport)(char *, size_t);
    void (*loudStub)(id, SEL);
};

unsigned gFailures;

void failure(const std::string &text)
{
    gFailures++;
    printf("FAIL %s\n", text.c_str());
}

template <typename T> T symbol(void *handle, const char *name)
{
    void *address = dlsym(handle, name);
    if (!address) {
        fprintf(stderr, "the shim has no %s\n", name);
        exit(2);
    }
    return (T)address;
}

Shim loadShim(const char *path)
{
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
        exit(2);
    }
    return {symbol<id<MTLDevice> (*)(uint64_t)>(handle, "WebMetalHostShimRegister"),
            symbol<int (*)(const LayoutSet *, char *, size_t)>(handle, "WebMetalHostShimCheckLayouts"),
            symbol<const LayoutSet *(*)(void)>(handle, "WebMetalHostShimExpectedLayouts"),
            symbol<size_t (*)(void)>(handle, "WMReportCount"), symbol<void (*)(char *, size_t)>(handle, "WMLastReport"),
            symbol<void (*)(id, SEL)>(handle, "WMLoudStub")};
}

std::string lastReport(const Shim &shim)
{
    char text[1024];
    shim.lastReport(text, sizeof text);
    return text;
}

const Role &deviceRole()
{
    for (const Role &role : target::kRoles)
        if (!strcmp(role.targetClass, "AGXA10XDevice"))
            return role;
    fprintf(stderr, "the generated tables have no AGXA10XDevice role\n");
    exit(2);
}

const BaseLayout &base(const char *name)
{
    for (const BaseLayout &layout : target::kBases)
        if (!strcmp(layout.name, name))
            return layout;
    fprintf(stderr, "the generated tables have no base %s\n", name);
    exit(2);
}

bool isDriverSelector(const char *name)
{
    const Role &role = deviceRole();
    for (size_t i = 0; i < role.count; i++)
        if (role.selectors[i].kind == SelectorKind::Instance && !strcmp(role.selectors[i].name, name))
            return true;
    return false;
}

// The answer of a zero- or one-integer-argument query, as a number. Returns false for a return type it cannot read.
bool ask(id device, SEL selector, const NSNumber *argument, NSNumber **answer)
{
    Method method = class_getInstanceMethod(object_getClass(device), selector);
    if (!method)
        return false;
    char type[256];
    method_getReturnType(method, type, sizeof type);
    const unsigned long long a = argument.unsignedLongLongValue;
    const bool one = argument != nil;
#define CALL(T) (one ? ((T(*)(id, SEL, unsigned long long))objc_msgSend)(device, selector, a) \
                     : ((T(*)(id, SEL))objc_msgSend)(device, selector))
    switch (type[0]) {
    case 'B': *answer = @(CALL(bool)); return true;
    case 'c': *answer = @(CALL(signed char)); return true;
    case 'C': *answer = @(CALL(unsigned char)); return true;
    case 's': *answer = @(CALL(short)); return true;
    case 'S': *answer = @(CALL(unsigned short)); return true;
    case 'i': *answer = @(CALL(int)); return true;
    case 'I': *answer = @(CALL(unsigned)); return true;
    case 'q': case 'l': *answer = @(CALL(long long)); return true;
    case 'Q': case 'L': *answer = @(CALL(unsigned long long)); return true;
    case 'f': *answer = @(CALL(float)); return true;
    case 'd': *answer = @(CALL(double)); return true;
    case '{':
        if (!strcmp(type, "{?=QQQ}")) {
            MTLSize size = ((MTLSize(*)(id, SEL))objc_msgSend)(device, selector);
            *answer = (NSNumber *)@[@(size.width), @(size.height), @(size.depth)];
            return true;
        }
        if (strstr(type, "=b")) {  // a structure of bit-fields: its bits as a little-endian integer
            unsigned width = 0;
            for (const char *p = strchr(type, '=') + 1; *p == 'b'; p++)
                width += (unsigned)strtoul(p + 1, (char **)&p, 10), p--;
            if (width == 32) {
                *answer = @(CALL(uint32_t));
                return true;
            }
        }
        return false;
    default:
        return false;
    }
#undef CALL
}

bool same(id answer, id expected)
{
    if ([answer isKindOfClass:[NSArray class]] || [expected isKindOfClass:[NSArray class]])
        return [answer isEqual:expected];
    NSNumber *a = answer, *e = expected;
    if (!strcmp(e.objCType, "d") || !strcmp(e.objCType, "f"))
        return a.doubleValue == e.doubleValue;
    if (e.objCType[0] == 'c' || e.objCType[0] == 'B')  // JSON booleans
        return a.boolValue == e.boolValue && a.unsignedLongLongValue <= 1;
    return a.unsignedLongLongValue == e.unsignedLongLongValue;
}

// Field offsets of the 21H461 _limits structure, from its instance-variable type encoding.
std::vector<std::pair<std::string, std::pair<ptrdiff_t, char>>> limitFields(const BaseIvar &limits)
{
    std::vector<std::pair<std::string, std::pair<ptrdiff_t, char>>> fields;
    const char *p = strchr(limits.type, '=') + 1;
    ptrdiff_t at = limits.offset;
    while (*p == '"') {
        const char *end = strchr(p + 1, '"');
        std::string name(p + 1, end);
        const char type = end[1];
        const ptrdiff_t size = type == 'Q' ? 8 : 4;
        at = (at + size - 1) / size * size;
        fields.push_back({name, {at, type}});
        at += size;
        p = end + 2;
    }
    return fields;
}

void checkLimits(id device, NSArray *expected, uint64_t sentinel)
{
    const BaseLayout &mtlDevice = base("_MTLDevice");
    const BaseIvar *limits = nullptr;
    for (size_t i = 0; i < mtlDevice.ivarCount; i++)
        if (!strcmp(mtlDevice.ivars[i].name, "_limits"))
            limits = &mtlDevice.ivars[i];
    Ivar ivar = class_getInstanceVariable(objc_getClass("_MTLDevice"), "_limits");
    if (!limits || !ivar || ivar_getOffset(ivar) != limits->offset || strcmp(ivar_getTypeEncoding(ivar), limits->type)) {
        failure("the _limits structure of this Mac's _MTLDevice is not 21H461's; its fields cannot be compared");
        return;
    }
    const uint8_t *object = (const uint8_t *)(__bridge void *)device;
    unsigned compared = 0;
    for (const auto &field : limitFields(*limits)) {
        const ptrdiff_t offset = field.second.first;
        const char type = field.second.second;
        NSDictionary *want = nil;
        for (NSDictionary *entry in expected)
            if ([entry[@"name"] isEqualToString:@(field.first.c_str())])
                want = entry;
        if (!want) {
            failure("_limits." + field.first + " is not in the device profile");
            continue;
        }
        if ([want[@"offset"] longLongValue] != offset) {
            failure("_limits." + field.first + " is at +" + std::to_string(offset) + " in the 21H461 encoding, the profile says +" +
                    std::to_string([want[@"offset"] longLongValue]));
            continue;
        }
        double value;
        unsigned long long integer = 0;
        if (type == 'f') {
            float f;
            memcpy(&f, object + offset, 4);
            value = f;
        } else if (type == 'Q') {
            memcpy(&integer, object + offset, 8);
            value = (double)integer;
        } else {
            uint32_t u;
            memcpy(&u, object + offset, 4);
            integer = u;
            value = u;
        }
        id runtime = want[@"runtime"];
        if (runtime) {
            if (integer != sentinel)
                failure("_limits." + field.first + " is " + std::to_string(integer) + ", the kernel input was " +
                        std::to_string(sentinel));
        } else if (type == 'f' ? value != [want[@"value"] doubleValue] : integer != [want[@"value"] unsignedLongLongValue]) {
            failure("_limits." + field.first + " is " + std::to_string(value) + ", the profile records " +
                    [[want[@"value"] description] UTF8String]);
        }
        compared++;
    }
    printf("limits: %u fields of _limits equal the device profile (maxBufferLength carries the sentinel 0x%llx)\n",
           compared, (unsigned long long)sentinel);
}

void checkAnswers(id device, NSArray *entries, const char *what)
{
    unsigned compared = 0, host = 0, skipped = 0;
    for (NSDictionary *entry in entries) {
        NSString *selectorName = entry[@"selector"];
        SEL selector = NSSelectorFromString(selectorName);
        NSNumber *argument = entry[@"argument"];
        // The plugin's own answers are compared always; Metal.framework's (inherited, or reached through the driver
        // getter as the target's does: "via") only where the profile's second method found this Mac's framework to
        // give the target's answer.
        const bool own = isDriverSelector(selectorName.UTF8String) && !entry[@"via"];
        if (!own && ![entry[@"b1"] boolValue]) {
            skipped++;
            continue;
        }
        NSNumber *answer = nil;
        if (!ask(device, selector, argument, &answer)) {
            failure(std::string(what) + " " + selectorName.UTF8String + ": cannot read its answer");
            continue;
        }
        if (!same(answer, entry[@"value"]))
            failure(std::string(what) + " " + selectorName.UTF8String + (argument ? "(" + std::string(argument.stringValue.UTF8String) + ")" : "") +
                    " answers " + answer.description.UTF8String + ", the profile records " +
                    [entry[@"value"] description].UTF8String);
        compared++;
        host += !own;
    }
    printf("%s: %u answers equal the device profile (%u of them Metal.framework's own); %u not compared: this Mac's "
           "framework answers them differently (the profile records why)\n", what, compared, host, skipped);
}

void checkSurface(id device, const Shim &shim)
{
    Class cls = object_getClass(device);
    Class mtlDevice = class_getSuperclass(cls);
    const Role &role = deviceRole();
    unsigned own = 0, stubs = 0;
    for (size_t i = 0; i < role.count; i++) {
        const DriverSelector &row = role.selectors[i];
        Class owner = row.kind == SelectorKind::Class ? object_getClass(cls) : cls;
        unsigned count = 0;
        Method *methods = class_copyMethodList(owner, &count);
        Method found = nullptr;
        for (unsigned k = 0; k < count; k++)
            if (!strcmp(sel_getName(method_getName(methods[k])), row.name))
                found = methods[k];
        free(methods);
        if (!found) {
            failure(std::string("driver selector ") + row.name + " is answered by neither the plugin nor a stub");
            continue;
        }
        if (strcmp(method_getTypeEncoding(found), row.types))
            failure(std::string(row.name) + " has encoding " + method_getTypeEncoding(found) + ", 21H461 has " + row.types);
        if (method_getImplementation(found) == (IMP)shim.loudStub)
            stubs++;
        else
            own++;
    }
    printf("surface: %zu driver selectors answered: %u by the plugin, %u by LOUD stubs\n", role.count, own, stubs);
    // Selectors the driver inherits from Metal.framework, or that no layer implements, must reach the base unchanged.
    const BaseLayout &layout = base(role.base);
    unsigned inherited = 0;
    for (size_t i = 0; i < layout.selectorCount; i++) {
        const BaseSelector &s = layout.selectors[i];
        if (s.kind != SelectorKind::Instance || isDriverSelector(s.name))
            continue;
        SEL selector = sel_registerName(s.name);
        if (class_getMethodImplementation(cls, selector) != class_getMethodImplementation(mtlDevice, selector))
            failure(std::string("WebMetalDevice overrides ") + s.name + ", which AGXA10XDevice does not");
        inherited++;
    }
    printf("surface: %u inherited or absent selectors reach _MTLDevice unchanged\n", inherited);
}

void checkLoud(id device, const Shim &shim)
{
    const size_t before = shim.reportCount();
    @try {
        (void)[device newCommandQueue];
        failure("-newCommandQueue returned instead of failing");
    } @catch (NSException *exception) {
        if (![exception.name isEqualToString:@"WebMetalNotImplemented"] ||
            ![exception.reason containsString:@"-[WebMetalDevice newCommandQueue]: not implemented"])
            failure(std::string("-newCommandQueue raised ") + exception.name.UTF8String + ": " + exception.reason.UTF8String);
    }
    if (shim.reportCount() != before + 1 || lastReport(shim).find("-[WebMetalDevice newCommandQueue]") == std::string::npos)
        failure("-newCommandQueue made no report: " + lastReport(shim));
    @try {
        (void)[device supportsVertexAmplificationCount:17];
        failure("-supportsVertexAmplificationCount:17 answered outside the profiled arguments");
    } @catch (NSException *exception) {
        if (![exception.reason containsString:@"with argument 17"])
            failure(std::string("-supportsVertexAmplificationCount:17 raised ") + exception.reason.UTF8String);
    }
    printf("loud: a stub and an argument outside the profile report and raise WebMetalNotImplemented (%zu reports)\n",
           shim.reportCount() - before);
}

void checkGuard(const Shim &shim)
{
    char text[512];
    if (!shim.checkLayouts(shim.expected(), text, sizeof text))
        failure(std::string("the guard refuses the layouts the shim was built with: ") + text);
    const LayoutSet target21H461 = {"iPadOS 17.7.11 (21H461)", target::kBases, sizeof target::kBases / sizeof *target::kBases};
    if (shim.checkLayouts(&target21H461, text, sizeof text))
        failure("the guard accepts 21H461's layouts on this Mac");
    else
        printf("guard: 21H461's layouts refused here: %s\n", text);
    // Alterations of the accepted layouts: each must be refused.
    const LayoutSet &host = *shim.expected();
    std::vector<BaseLayout> bases(host.bases, host.bases + host.count);
    size_t device = 0;
    while (strcmp(bases[device].name, "_MTLDevice"))
        device++;
    const BaseLayout original = bases[device];
    std::vector<BaseSelector> selectors(original.selectors, original.selectors + original.selectorCount);
    std::vector<BaseIvar> ivars(original.ivars, original.ivars + original.ivarCount);
    const char *names[] = {"instance size", "instance variable offset", "selector encoding", "selector presence",
                           "superclass"};
    for (int alteration = 0; alteration < 5; alteration++) {
        BaseLayout altered = original;
        std::vector<BaseSelector> s = selectors;
        std::vector<BaseIvar> v = ivars;
        size_t present = 0;
        while (!s[present].types)
            present++;
        switch (alteration) {
        case 0: altered.instanceSize += 8; break;
        case 1: v[0].offset += 8; altered.ivars = v.data(); break;
        case 2: s[present].types = "v16@0:8"; altered.selectors = s.data(); break;
        case 3: s[present].types = nullptr; altered.selectors = s.data(); break;
        case 4: altered.superclass = "_MTLObjectWithLabel"; break;
        }
        std::vector<BaseLayout> set = bases;
        set[device] = altered;
        const LayoutSet changed = {"altered", set.data(), set.size()};
        if (shim.checkLayouts(&changed, text, sizeof text))
            failure(std::string("the guard accepts a changed ") + names[alteration]);
    }
    printf("guard: this Mac's recorded layouts accepted; 5 alterations of them refused\n");
}

}  // namespace

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s register|strict|refuse shim.dylib [expectations.json sentinel]\n", argv[0]);
            return 2;
        }
        const std::string mode = argv[1];
        const Shim shim = loadShim(argv[2]);
        if (mode == "strict") {
            id<MTLDevice> device = shim.registerDevice(0x5e5e1000);
            if (!device)
                return 3;
            (void)[device newCommandQueue];
            printf("FAIL strict mode returned from a LOUD stub\n");
            return 1;
        }
        if (mode == "refuse") {
            id<MTLDevice> device = shim.registerDevice(0x5e5e1000);
            const std::string report = lastReport(shim);
            if (device || report.find("refused: the running Metal.framework is not the one the plugin was built for "
                                      "(iPadOS 17.7.11 (21H461))") == std::string::npos) {
                printf("FAIL a shim built for 21H461's layouts created a device here (report: %s)\n", report.c_str());
                return 1;
            }
            printf("PASS: the plugin built for 21H461's layouts refuses this Mac's Metal.framework: %s\n", report.c_str());
            return 0;
        }
        if (mode != "register" || argc != 5) {
            fprintf(stderr, "unknown mode %s\n", argv[1]);
            return 2;
        }
        NSData *data = [NSData dataWithContentsOfFile:@(argv[3])];
        NSDictionary *expectations = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
        if (!expectations) {
            fprintf(stderr, "cannot read %s\n", argv[3]);
            return 2;
        }
        const uint64_t sentinel = strtoull(argv[4], nullptr, 0);
        id<MTLDevice> device = shim.registerDevice(sentinel);
        if (!device) {
            printf("FAIL the shim created no device: %s\n", lastReport(shim).c_str());
            return 1;
        }
        bool listed = [MTLCopyAllDevices() containsObject:device];
        if (!listed)
            failure("MTLCopyAllDevices does not list the device");
        if (![device conformsToProtocol:@protocol(MTLDevice)] || ![device conformsToProtocol:objc_getProtocol("MTLDeviceSPI")])
            failure("the device does not conform to MTLDevice and MTLDeviceSPI");
        printf("register: MTLAddDevice accepted %s and MTLCopyAllDevices lists it (registryID 0x%llx)\n",
               class_getName(object_getClass(device)), (unsigned long long)device.registryID);
        checkLimits(device, expectations[@"limits"], sentinel);
        checkAnswers(device, expectations[@"getters"], "getters");
        checkAnswers(device, expectations[@"queries"], "queries");
        checkSurface(device, shim);
        checkLoud(device, shim);
        checkGuard(shim);
        printf("%s: %u failures\n", gFailures ? "FAIL" : "PASS", gFailures);
        return gFailures ? 1 : 0;
    }
}
