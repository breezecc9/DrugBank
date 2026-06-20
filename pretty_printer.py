import os
import re
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from custom_printer import ptr_color


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
    def __init__(self, value: str = "") -> None:
        super().__init__()
        self._value = value

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


class BaseValueTemplate(Generic[T]):
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
        color: Optional[str] = None,
        length: Optional[int] = None,
    ):
        if color is not None:
            self.set_color(color)
        if length is not None:
            self.set_length(length)
        self.set_value(value)

    def assemble(self) -> str:
        template = f"{{lpad}}{self._color}{{value}}\033[0m{{rpad}}"
        template_dict = {"lpad": "", "rpad": ""}
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
        color: Optional[str] = None,
        length: Optional[int] = None,
    ) -> str:
        self.write(value, color, length)
        return self.assemble()

    def __len__(self) -> int:
        return self._length


class ScrollTemplate:
    def __init__(self) -> None:
        self.pos = 0
        self._value_index_list: list[int] = []
        self.length = 0

    def add_template(self, index: int):
        self._value_index_list.append(index)
        self.length += 1


class PrettyPrinter:
    def __init__(self) -> None:
        self._template: list[BaseValueTemplate[Any]] = []
        self._template_dict: dict[str, BaseValueTemplate[Any]] = {}
        self._scroll_dict: dict[str, ScrollTemplate] = {}

    def add_template(
        self,
        key: Optional[str],
        template: BaseValue[T],
        color: str = PtrColor.info,
        length: int = 30,
    ):
        wrapper = BaseValueTemplate(template, color, length)
        self._template.append(wrapper)
        if key is not None:
            self._template_dict[key] = wrapper
        return self

    def add_string_template(
        self,
        value: str = "",
        color: str = PtrColor.info,
        length: int = 30,
        key: Optional[str] = None,
    ):
        return self.add_template(key, StringValue(value), color, length)

    def next_line(
        self,
    ):
        return self.add_template(None, StringValue("\n"), ptr_color.info, 2)

    def split_line(self, length: int = 150, char: str = "-", front=True, back=True):
        if front:
            self.next_line()
        self.add_template(
            None, StringValue(char * length), ptr_color.info, length * len(char)
        )
        if back:
            self.next_line()
        return self

    def add_placeholder_template(
        self,
        placeholders: str,
        value: Optional[dict[str, str]] = None,
        color: str = PtrColor.info,
        length: int = 30,
        key: Optional[str] = None,
    ):
        template = PlaceholderValue(placeholders)
        if value is not None:
            template.write(value)
        return self.add_template(key, template, color, length)

    def write(
        self,
        key: str,
        value: str | dict[str, str],
        color: Optional[str] = None,
        length: Optional[int] = None,
    ):

        tpl = self._template_dict.get(key)
        if tpl is None:
            raise KeyError(f"不存在key={key}的模板")
        tpl.write(value, color, length)

    def w_flush(
        self,
        key: str,
        value: str | dict[str, str],
        color: Optional[str] = None,
        length: Optional[int] = None,
    ):
        self.write(key, value, color, length)
        self.flush()

    def flush(self):
        self.clear_screen()
        print("".join([v.assemble() for v in self._template]))

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def add_scroll(self, key: str):
        current_scroll = self._scroll_dict.get(key)
        if current_scroll is None:
            current_scroll = ScrollTemplate()
            self._scroll_dict[key] = current_scroll
        current_scroll.add_template(len(self._template) - 1)
        return self

    def scroll(
        self,
        key: str,
        value: str | dict[str, str],
        color: str | None = None,
        length: int | None = None,
    ):
        current_scroll = self._scroll_dict.get(key)
        if current_scroll is None:
            return
        if current_scroll.length <= 0:
            return
        if current_scroll.pos < current_scroll.length:
            self._template[current_scroll._value_index_list[current_scroll.pos]].write(
                value, color, length
            )
            current_scroll.pos += 1
        else:
            template_list: list[BaseValueTemplate[Any]] = []
            for i in current_scroll._value_index_list:
                template_list.append(self._template[i])
            current_template = template_list.pop(0)
            current_template.write(value, color, length)
            template_list.append(current_template)
            for i, j in enumerate(current_scroll._value_index_list):
                self._template[j] = template_list[i]

    def scl_flush(
        self,
        key: str,
        value: str | dict[str, str],
        color: str | None = None,
        length: int | None = None,
    ):
        self.scroll(key, value, color, length)
        self.flush()


pt_printer = PrettyPrinter()
pt_printer.split_line().add_placeholder_template(
    "Name:{value}", key="name"
).add_placeholder_template("Encoder:{value}", key="encoder").add_placeholder_template(
    "Classifier:{value}", key="classifier"
).add_placeholder_template("LR:{value}", key="lr").add_placeholder_template(
    "Device:{value}", key="device"
).split_line().add_placeholder_template(
    "Epoch:{value}", key="epoch"
).add_placeholder_template("Elapsed:{value}", key="elapsed").add_placeholder_template(
    "Early Stop:{value}", key="early_stop"
).add_placeholder_template("State:{value}", key="state").add_placeholder_template(
    "Stage:{value}", key="stage"
).split_line().add_string_template("Train:").add_placeholder_template(
    "Loss:{loss} Acc:{acc}", length=120, key="train"
).split_line().add_string_template("Last Val:").add_placeholder_template(
    "Loss:{loss} Acc:{acc} F1:{f1} Prec:{prec} Rec:{rec}", length=120, key="val"
).split_line().add_string_template("Test:").add_placeholder_template(
    "Loss:{loss} Acc:{acc} F1:{f1} Prec:{prec} Rec:{rec}", length=120, key="test"
).split_line().add_string_template().add_string_template(length=120).add_scroll(
    "info"
).next_line().add_string_template().add_string_template(length=120).add_scroll(
    "info"
).next_line().add_string_template("Info").add_string_template(length=120).add_scroll(
    "info"
).next_line().add_string_template().add_string_template(length=120).add_scroll(
    "info"
).next_line().add_string_template().add_string_template(length=120).add_scroll(
    "info"
).split_line()


__all__ = ["pt_printer", "PtrColor"]
