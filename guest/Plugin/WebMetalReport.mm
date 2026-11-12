// The plugin's not-implemented channel (WebMetalReport.h).

#import "WebMetalReport.h"
#import "WebMetalSurface.h"

#import <objc/runtime.h>
#include <os/log.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

NSString *const WMNotImplementedException = @"WebMetalNotImplemented";

namespace {

std::mutex gLock;
size_t gCount;
std::string gLast;

bool strictMode()
{
    static const bool strict = [] {
        const char *value = getenv("WEBMETAL_STRICT");
        return value && strcmp(value, "1") == 0;
    }();
    return strict;
}

os_log_t reportLog()
{
    static os_log_t log = os_log_create("com.unixnu.WebMetal", "not-implemented");
    return log;
}

std::string methodName(id self, SEL selector)
{
    Class cls = object_getClass(self);
    const bool meta = class_isMetaClass(cls);
    return std::string(meta ? "+[" : "-[") + class_getName(cls) + " " + sel_getName(selector) + "]";
}

// The generated table row for a stubbed selector of the receiver's plugin class, if there is one.
const webmetal::DriverSelector *driverRow(id self, SEL selector, const webmetal::Role **owner)
{
    Class cls = object_getClass(self);
    const bool meta = class_isMetaClass(cls);
    const webmetal::SelectorKind kind = meta ? webmetal::SelectorKind::Class : webmetal::SelectorKind::Instance;
    size_t count = 0;
    const webmetal::Role *roles = webmetal::Roles(&count);
    for (Class c = meta ? (Class)self : cls; c; c = class_getSuperclass(c)) {
        for (size_t i = 0; i < count; i++) {
            if (strcmp(class_getName(c), roles[i].pluginClass) != 0)
                continue;
            for (size_t k = 0; k < roles[i].count; k++) {
                const webmetal::DriverSelector &row = roles[i].selectors[k];
                if (row.kind == kind && strcmp(row.name, sel_getName(selector)) == 0) {
                    *owner = &roles[i];
                    return &row;
                }
            }
        }
    }
    return nullptr;
}

__attribute__((noreturn)) void fail(const std::string &what, const std::string &detail)
{
    const std::string problem = detail.empty() ? "not implemented" : "not implemented: " + detail;
    WMReport(what.c_str(), problem.c_str());
    @throw [NSException exceptionWithName:WMNotImplementedException
                                   reason:[NSString stringWithFormat:@"%s: %s", what.c_str(), problem.c_str()]
                                 userInfo:nil];
}

}  // namespace

void WMReport(const char *what, const char *problem)
{
    const std::string line = std::string("WebMetal: ") + what + ": " + problem;
    {
        std::lock_guard<std::mutex> guard(gLock);
        gCount++;
        gLast = line;
    }
    fprintf(stderr, "%s\n", line.c_str());
    os_log_fault(reportLog(), "%{public}s", line.c_str());
    if (strictMode()) {
        fprintf(stderr, "WebMetal: WEBMETAL_STRICT: stopping at the first report\n");
        abort();
    }
}

void WMLoudMethod(id self, SEL _cmd, const char *detail)
{
    fail(methodName(self, _cmd), detail ? detail : "");
}

void WMLoudArgument(id self, SEL _cmd, uint64_t argument, const char *detail)
{
    fail(methodName(self, _cmd) + " with argument " + std::to_string(argument), detail ? detail : "");
}

void WMLoudStub(id self, SEL _cmd)
{
    const webmetal::Role *role = nullptr;
    const webmetal::DriverSelector *row = driverRow(self, _cmd, &role);
    std::string detail;
    if (row) {
        detail = std::string("21H461 ") + role->targetClass + " implements it in its " + row->layer + " layer";
        if (*row->detail)
            detail += std::string("; ") + row->detail;
    }
    fail(methodName(self, _cmd), detail);
}

size_t WMReportCount(void)
{
    std::lock_guard<std::mutex> guard(gLock);
    return gCount;
}

void WMLastReport(char *buffer, size_t size)
{
    std::lock_guard<std::mutex> guard(gLock);
    if (size)
        snprintf(buffer, size, "%s", gLast.c_str());
}
