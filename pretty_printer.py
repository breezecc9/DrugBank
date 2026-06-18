import re
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar


class PtrColor:
    notice: str = "\033[38;2;150;150;255m"
    flag: str = "\033[38;2;0;255;0m"
    info: str = "\033[38;2;255;255;255m"
    warning: str = "\033[38;2;255;165;0m"
    error: str = "\033[38;2;255;0;0m"

    pending: str = "\033[38;2;255;255;150m"
    training: str = "\033[38;2;150;255;150m"
    validating: str = "\033[38;2;0;255;255m"

    done: str = "\033[38;2;128;128;128m"


def get_all_placeholder(template: str) -> list[str]:
    pattern = re.compile(r"\{(.+?)\}")
    placeholders = pattern.findall(template)
    return placeholders


T = TypeVar("T")


class BaseValue(ABC, Generic[T]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def write(self, value: T):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def value(self) -> str:
        pass


class StringValue(BaseValue[str]):
    def __init__(self) -> None:
        super().__init__()
        self._value = ""

    def write(self, value):
        self._value = value

    def __len__(self):
        return len(self._value)

    @property
    def value(self) -> str:
        return self._value


class PlaceholderValue(BaseValue[dict[str, str]]):
    def __init__(self, template: str) -> None:
        super().__init__()
        self._template = template
        self._placeholders: dict[str, str] = {
            ph: "" for ph in get_all_placeholder(template)
        }
        self._value = template.format(**self._placeholders)

    def write(self, value):
        invalid_keys = [k for k in value if k not in self._placeholders]
        if invalid_keys:
            raise KeyError(f"invalid_keys: {invalid_keys}")
        self._placeholders.update(value)
        self._value = self._template.format(**self._placeholders)

    def __len__(self):
        return len(self._value)

    @property
    def value(self):
        return self._value


class BaseValueTemplate(ABC, Generic[T]):
    def __init__(
        self, value: BaseValue[T], color: str = PtrColor.info, length: int = 30
    ) -> None:
        self._value_template = value
        self.set_length(length)
        self.set_color(color)

    def set_length(self, length: int):
        self._length = length

    def set_color(self, color: str):
        self._color = color

    def set_value(self, value: T):
        self._value_template.write(value)

    def write(
        self,
        value: T,
        color: str | None = None,
        length: int | None = None,
    ):
        if color is not None:
            self.set_color(color)
        if length is not None:
            self.set_length(length)
        self.set_value(value)

    def assemble(self) -> str:
        template = "{lpad}{prefix}{value}{suffix}{rpad}"
        template_dict = {"lpad": "", "rpad": ""}
        template_dict["prefix"] = self._color
        template_dict["suffix"] = PtrColor.info
        diff = self._length - len(self._value_template.value)
        if diff < 0:
            keep_len = max(self._length - 3, 0)
            template_dict["value"] = self._value_template.value[0:keep_len] + "..."
        else:
            template_dict["value"] = self._value_template.value
            template_dict["lpad"] = (diff // 2) * " "
            template_dict["rpad"] = (diff - diff // 2) * " "
        return template.format(**template_dict)

    def w_assemble(
        self,
        value: T,
        color: str | None = None,
        length: int | None = None,
    ) -> str:
        self.write(value, color, length)
        return self.assemble()

    def __len__(self) -> int:
        return self._length


value_template_index = {0: StringValue, 1: PlaceholderValue}


class PrettyPrinter:
    def __init__(self) -> None:
        self._tempalte = ""
        self._template_dict: dict[str, BaseValueTemplate] = {}

    def add_template(self, template: str):
        placeholders = get_all_placeholder(template)
        for placeholder in placeholders:
            arr = placeholder.split(":")
            if len(arr) < 4:
                raise ValueError(f"{placeholder} error")
            self._template_dict[arr[0]] = BaseValueTemplate(
                StringValue(), color=arr[1], length=int(arr[2])
            )
            template.replace(placeholder, arr[0])
        self._tempalte += template

    def flush(self):
        flush_dict = {}
        for k, v in self._template_dict.items():
            flush_dict[k] = v.assemble()


x = "{op:xx:12}{op1::12}"
pattern = re.compile(r"\{(.+?)\}")
placeholders = pattern.findall(x)
for placeholder in placeholders:
    print(placeholder.split(":"))

