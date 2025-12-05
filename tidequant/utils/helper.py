"""
辅助编写代码相关的辅助工具
"""

class DummyClass:
    """
    作为占位符的类, 执行任何方法都不报错
    """

    def __getattr__(self, name: str) -> "DummyClass":
        return self
    
    def __call__(self, *args, **kwargs) -> None:
        pass
