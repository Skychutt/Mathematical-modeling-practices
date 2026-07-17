"""
数学建模 · Python 基础 05：文件读写
竞赛中常见：读 csv/txt 数据，写出结果表。
"""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "_demo_data"


def write_demo_files():
    DATA_DIR.mkdir(exist_ok=True)

    # 文本
    txt_path = DATA_DIR / "notes.txt"
    txt_path.write_text("第一行\n第二行\n", encoding="utf-8")

    # CSV（用纯文本即可，正式项目可用 pandas / csv 模块）
    csv_path = DATA_DIR / "points.csv"
    lines = ["x,y,label", "0,0,A", "1,2,B", "2,1,C"]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path, csv_path


def read_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_csv_simple(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = read_text(path)
    header = rows[0].split(",")
    data = [r.split(",") for r in rows[1:]]
    return header, data


def append_result(path: Path, line: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    txt, csv = write_demo_files()
    print("文本内容:", read_text(txt))
    header, data = read_csv_simple(csv)
    print("表头:", header)
    print("数据:", data)

    out = DATA_DIR / "result.txt"
    out.write_text("iter,best\n", encoding="utf-8")
    append_result(out, "1,12.3")
    append_result(out, "2,10.1")
    print("结果文件:\n" + out.read_text(encoding="utf-8"))
