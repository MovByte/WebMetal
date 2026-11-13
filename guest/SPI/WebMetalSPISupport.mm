// Process facts the driver's SPI reads (WebMetalSPI.h).

#import "WebMetalSPI.h"

#import <CoreFoundation/CoreFoundation.h>

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <string>

// Security.framework's task entitlement queries (Security/SecTask.h, which the iPhoneOS SDK does not ship).
typedef struct CF_BRIDGED_TYPE(id) __SecTask *SecTaskRef;
extern "C" SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef allocator);
extern "C" CFTypeRef SecTaskCopyValueForEntitlement(SecTaskRef task, CFStringRef entitlement, CFErrorRef *error);

namespace webmetal {

bool InternalBuild(id self, SEL _cmd)
{
    // libMobileGestalt has no header in the SDK; the driver links it and calls MGGetBoolAnswer with the key.
    static const int internal = [] {
        void *library = dlopen("/usr/lib/libMobileGestalt.dylib", RTLD_LAZY | RTLD_LOCAL);
        auto answer = library ? (bool (*)(CFStringRef))dlsym(library, "MGGetBoolAnswer") : nullptr;
        return answer ? int(answer(CFSTR("InternalBuild"))) : -1;
    }();
    if (internal < 0)
        WMLoudMethod(self, _cmd, "libMobileGestalt's MGGetBoolAnswer is not in this process, so the driver's "
                                 "internal-build answer is unknown");
    return internal;
}

bool TaskHasEntitlement(const char *name)
{
    static std::mutex lock;
    static std::map<std::string, bool> answers;
    std::lock_guard<std::mutex> guard(lock);
    auto found = answers.find(name);
    if (found != answers.end())
        return found->second;
    bool entitled = false;
    if (SecTaskRef task = SecTaskCreateFromSelf(kCFAllocatorDefault)) {
        CFStringRef key = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
        CFTypeRef value = SecTaskCopyValueForEntitlement(task, key, nullptr);
        entitled = value == kCFBooleanTrue;
        if (value)
            CFRelease(value);
        CFRelease(key);
        CFRelease(task);
    }
    answers[name] = entitled;
    return entitled;
}

}  // namespace webmetal
