"""
    修复IDEA模块识别
"""


def template_modules_xml(main_module, sub_modules: list):
    str_modules = "\n      ".join([
        f'<module'
        f' fileurl="file://$PROJECT_DIR$/modules/{item}/{item}.iml"'
        f' filepath="$PROJECT_DIR$/modules/{item}/{item}.iml" />'
        for item in sub_modules
    ])
    return f"""
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://$PROJECT_DIR$/.idea/{main_module}.iml" filepath="$PROJECT_DIR$/.idea/{main_module}.iml" />
      {str_modules} 
    </modules>
  </component>
</project>
    """.strip()


def template_module_iml():
    return f"""
<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager" inherit-compiler-output="true">
    <exclude-output />
    <content url="file://$MODULE_DIR$" />
    <orderEntry type="inheritedJdk" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
    """.strip()


if __name__ == '__main__':
    import os

    buffer_modules = []
    for module_name in os.listdir("./modules"):
        buffer_modules.append(module_name)
        # IML
        with open(f"./modules/{module_name}/{module_name}.iml", 'w') as fp:
            fp.write(template_module_iml())
    print(buffer_modules)

    # main.iml
    MAIN = "HelloML"
    with open(f"./.idea/{MAIN}.iml", 'w') as fp:
        fp.write(template_module_iml())

    # modules.xml
    with open("./.idea/modules.xml", 'w') as fp:
        fp.write(template_modules_xml(MAIN, buffer_modules))
