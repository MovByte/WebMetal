// LOUD stubs: every selector the 21H461 driver class implements in its own layers is answered by the plugin class that
// stands in for it, either with an implementation or with WMLoudStub. A selector missing from a plugin class would
// otherwise reach its Metal.framework base, whose default is not what the driver does.

#import "WebMetalSurface.h"
#import "WebMetalReport.h"

#include "../Generated/WebMetalTargetSurface.h"

#import <objc/runtime.h>

#include <cstring>

namespace webmetal {

const Role *Roles(size_t *count)
{
    *count = sizeof target::kRoles / sizeof *target::kRoles;
    return target::kRoles;
}

bool InstallLoudStubs(std::string *problem)
{
    for (const Role &role : target::kRoles) {
        Class cls = objc_getClass(role.pluginClass);
        if (!cls) {
            *problem = std::string("no class ") + role.pluginClass + " for " + role.targetClass;
            return false;
        }
        Class base = class_getSuperclass(cls);
        if (!base || strcmp(class_getName(base), role.base) != 0) {
            *problem = std::string(role.pluginClass) + " is not a direct subclass of " + role.base;
            return false;
        }
        for (size_t i = 0; i < role.count; i++) {
            const DriverSelector &row = role.selectors[i];
            Class owner = row.kind == SelectorKind::Class ? object_getClass(cls) : cls;
            SEL selector = sel_registerName(row.name);
            if (class_addMethod(owner, selector, (IMP)WMLoudStub, row.types))
                continue;
            // class_addMethod refuses only a selector the class implements itself: then its own method must have the
            // driver method's signature.
            unsigned count = 0;
            Method *methods = class_copyMethodList(owner, &count);
            const char *types = nullptr;
            for (unsigned k = 0; k < count && !types; k++)
                if (method_getName(methods[k]) == selector)
                    types = method_getTypeEncoding(methods[k]);
            free(methods);
            if (!types || strcmp(types, row.types) != 0) {
                *problem = std::string(row.kind == SelectorKind::Class ? "+[" : "-[") + role.pluginClass + " " +
                           row.name + "] has type encoding " + (types ? types : "?") + ", 21H461 " +
                           role.targetClass + " has " + row.types;
                return false;
            }
        }
    }
    return true;
}

}  // namespace webmetal
