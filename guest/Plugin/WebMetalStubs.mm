// LOUD stubs: every selector the 21H461 driver implements in the layers the plugin replaces is answered by the plugin's
// class that carries it, either with an implementation or with WMLoudStub. A selector missing from a plugin class would
// otherwise reach its Metal.framework base, whose default is not what the driver does.

#import "WebMetalSurface.h"
#import "WebMetalReport.h"

#include "../Generated/WebMetalTargetSurface.h"

#import <objc/runtime.h>

#include <dlfcn.h>

#include <cstring>

namespace webmetal {

const Role *Roles(size_t *count)
{
    *count = sizeof target::kRoles / sizeof *target::kRoles;
    return target::kRoles;
}

Class PluginClass(const char *name)
{
    Dl_info image;
    if (!dladdr(reinterpret_cast<const void *>(&PluginClass), &image) || !image.dli_fbase)
        return Nil;
    __block Class found = Nil;
    objc_enumerateClasses(image.dli_fbase, name, nil, Nil, ^(Class cls, BOOL *stop) {
        if (strcmp(class_getName(cls), name) == 0) {
            found = cls;
            *stop = YES;
        }
    });
    return found;
}

bool InstallLoudStubs(std::string *problem)
{
    for (const Role &role : target::kRoles) {
        Class cls = PluginClass(role.name);
        if (!cls) {
            *problem = std::string("no class ") + role.name;
            return false;
        }
        // Inside the bundle the superclass is the plugin's own class of that name; past it, the base.
        Class expected = strcmp(role.superclass, role.base) ? PluginClass(role.superclass) : objc_getClass(role.base);
        if (!expected || class_getSuperclass(cls) != expected) {
            *problem = std::string(role.name) + " is not a direct subclass of " + role.superclass;
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
                *problem = std::string(row.kind == SelectorKind::Class ? "+[" : "-[") + role.name + " " + row.name +
                           "] has type encoding " + (types ? types : "?") + ", 21H461 " + row.definer + " has " +
                           row.types;
                return false;
            }
        }
    }
    return true;
}

}  // namespace webmetal
