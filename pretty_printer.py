import re


class ptr_color:
    notice: str = "\033[38;2;150;150;255m"
    flag: str = "\033[38;2;0;255;0m"
    info: str = "\033[38;2;255;255;255m"
    warning: str = "\033[38;2;255;165;0m"
    error: str = "\033[38;2;255;0;0m"

    pending: str = "\033[38;2;255;255;150m"
    training: str = "\033[38;2;150;255;150m"
    validating: str = "\033[38;2;0;255;255m"

    done: str = "\033[38;2;128;128;128m"


class base_vlaue_template:
    def __init__(
        self, value: str = "", color: str = ptr_color.info, length: int = 30
    ) -> None:
        self.set_value(value)
        self.set_length(length)
        self.set_color(color)

    def set_value(self, value: str):
        self._value = value

    def set_length(self, length: int):
        self._length = length

    def set_color(self, color: str):
        if color == "":
            color = ptr_color.info
        self._color = color

    def write(
        self,
        value: str,
        color: str | None = None,
        length: int | None = None,
    ):
        self.set_value(value)
        if length is not None:
            self.set_length(length)
        if color is not None:
            self.set_color(color)

    def assemble(self) -> str:
        template = "{lpad}{prefix}{value}{suffix}{rpad}"
        template_dict = {"lpad": "", "rpad": ""}
        template_dict["prefix"] = self.color
        template_dict["suffix"] = self.color
        diff = self.length - len(self.value)
        if diff < 0:
            template_dict["value"] = self.value[0:-4] + "..."
        else:
            template_dict["value"] = self.value
            template_dict["lpad"] = (diff // 2) * " "
            template_dict["rpad"] = (diff - (diff // 2)) * " "
        return template.format(**template_dict)

    def w_assemble(
        self,
        value: str,
        color: str | None = None,
        length: int | None = None,
    ) -> str:
        self.write(value, color, length)
        return self.assemble()

    @property
    def value(self) -> str:
        return self._value

    @property
    def color(self) -> str:
        return self._color

    @property
    def length(self) -> int:
        return self._length

    def __len__(self) -> int:
        return self.length


class pretty_printer:
    def __init__(self) -> None:
        self._tempalte = ""
        self._template_dict: dict[str, base_vlaue_template] = {}

    def get_all_placeholder(self, template: str) -> list[str]:
        pattern = re.compile(r"\{(.+?)\}")
        placeholders = pattern.findall(template)
        return placeholders

    def add_template(self, template: str):
        placeholders = self.get_all_placeholder(template)
        for placeholder in placeholders:
            arr = placeholder.split(":")
            if len(arr) != 3:
                raise ValueError(f"{placeholder} error")
            self._template_dict[arr[0]] = base_vlaue_template(
                color=arr[1], length=int(arr[2])
            )
            template.replace(placeholder, arr[0])
        self._tempalte += template

    def flush(self):
        flush_dict = {}
        for k, v in self._template_dict.items():
            flush_dict[k] = v.assemble()


# x = "{op:xx:12}{op1::12}"
# pattern = re.compile(r"\{(.+?)\}")
# placeholders = pattern.findall(x)
# for placeholder in placeholders:
#     print(placeholder.split(":"))
