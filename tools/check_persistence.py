#!/usr/bin/env python3
"""Validate the metrics persisted by a Mammoth smoke run.

Mammoth appends one Python-literal dictionary per run to ``logs.pyd``.  This
checker deliberately uses :func:`ast.literal_eval` (never ``eval``), selects
the requested run, and verifies that the information needed by the Pareto
study survived the complete logging path.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Any


REQUIRED_ARGUMENT_FIELDS = (
    "model",
    "dataset",
    "seed",
    "lr",
    "batch_size",
    "n_epochs",
    "fitting_mode",
)


class SafeLogLiteralNormalizer(ast.NodeTransformer):
    """Normalize the few non-literal reprs known to occur in ``vars(args)``.

    ``torch.device`` is rendered as ``device(type='cuda', index=0)`` and newer
    NumPy versions may render scalars as ``np.float64(1.0)``.  Neither is
    accepted by ``ast.literal_eval``.  We accept only these tightly constrained
    call shapes, replace them with plain literals, and then still hand the
    complete tree to ``ast.literal_eval``.  Arbitrary calls remain forbidden.
    """

    NUMPY_SCALAR_NAMES = {
        "bool_",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802 (AST API name)
        if isinstance(node.func, ast.Name) and node.func.id == "device":
            if node.args or any(keyword.arg not in {"type", "index"} for keyword in node.keywords):
                raise ValueError("repr de device no permitido")
            fields: dict[str, Any] = {}
            for keyword in node.keywords:
                require(keyword.arg is not None, "device no admite **kwargs en logs.pyd")
                fields[keyword.arg] = ast.literal_eval(self.visit(keyword.value))
            require(isinstance(fields.get("type"), str), "device.type debe ser una cadena")
            require(
                fields.get("index") is None or isinstance(fields.get("index"), int),
                "device.index debe ser entero o None",
            )
            suffix = "" if fields.get("index") is None else f":{fields['index']}"
            return ast.copy_location(ast.Constant(f"{fields['type']}{suffix}"), node)

        is_numpy_scalar = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"np", "numpy"}
            and node.func.attr in self.NUMPY_SCALAR_NAMES
        )
        if is_numpy_scalar:
            if len(node.args) != 1 or node.keywords:
                raise ValueError("repr de escalar NumPy no permitido")
            value_node = self.visit(node.args[0])
            # This proves that the scalar payload is itself only a literal.
            ast.literal_eval(value_node)
            return ast.copy_location(value_node, node)

        raise ValueError(f"llamada no permitida en logs.pyd: {ast.unparse(node.func)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 (AST API name)
        if node.id == "nan":
            return ast.copy_location(ast.Constant(float("nan")), node)
        if node.id == "inf":
            return ast.copy_location(ast.Constant(float("inf")), node)
        return node


def safe_literal_dict(text: str) -> dict[str, Any]:
    """Parse one logger line without executing any code."""
    tree = ast.parse(text, mode="eval")
    normalized = SafeLogLiteralNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    value = ast.literal_eval(normalized)
    require(isinstance(value, dict), f"El registro no es un diccionario: {type(value).__name__}")
    return value


def require(condition: bool, message: str) -> None:
    """Raise an assertion with a useful message for the server operator."""
    if not condition:
        raise AssertionError(message)


def read_records(path: Path, *, after_line: int = 0) -> list[tuple[int, dict[str, Any]]]:
    require(path.is_file(), f"No existe el fichero de resultados: {path}")

    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            # A retry must be isolated from every old record, including an old
            # malformed record.  Do not even parse the protected prefix.
            if line_number <= after_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = safe_literal_dict(line)
            except (AssertionError, SyntaxError, ValueError) as error:
                raise AssertionError(
                    f"La linea {line_number} de {path} no es un literal Python valido: {error}"
                ) from error
            records.append((line_number, value))

    require(records, f"No hay registros en {path} posteriores a la linea {after_line}.")
    return records


def select_record(
    records: list[tuple[int, dict[str, Any]]],
    *,
    model: str,
    dataset: str,
    seed: int,
    notes: str | None,
    after_line: int,
) -> tuple[int, dict[str, Any]]:
    """Return the newest record matching the smoke-run identity."""
    normalized_model = model.replace("_", "-")
    matching: list[tuple[int, dict[str, Any]]] = []
    for line_number, record in records:
        if line_number <= after_line:
            continue
        record_model = record.get("model")
        if not isinstance(record_model, str) or record_model.replace("_", "-") != normalized_model:
            continue
        if record.get("dataset") != dataset:
            continue
        try:
            record_seed = int(record.get("seed"))
        except (TypeError, ValueError):
            continue
        if record_seed != seed:
            continue
        if notes is not None and record.get("notes") != notes:
            continue
        matching.append((line_number, record))

    selector = f"model={model!r}, dataset={dataset!r}, seed={seed}"
    if notes is not None:
        selector += f", notes={notes!r}"
    require(
        matching,
        f"No hay ningun registro posterior a la linea {after_line} que coincida con {selector}.",
    )
    return matching[-1]


def validate_arguments(
    record: dict[str, Any],
    *,
    expected_lr: float,
    expected_batch_size: int,
) -> None:
    """Validate the flattened ``vars(args)`` fields written by Logger.write."""
    missing = [field for field in REQUIRED_ARGUMENT_FIELDS if field not in record]
    require(
        not missing,
        "Faltan campos de vars(args) en logs.pyd: " + ", ".join(missing),
    )

    try:
        actual_lr = float(record["lr"])
    except (TypeError, ValueError) as error:
        raise AssertionError(f"El LR persistido no es numerico: {record['lr']!r}") from error

    require(
        math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12),
        f"LR efectivo incorrecto: esperado {expected_lr:.12g}, encontrado {actual_lr:.12g}.",
    )
    require(
        int(record["batch_size"]) == expected_batch_size,
        f"Batch size incorrecto: esperado {expected_batch_size}, encontrado {record['batch_size']!r}.",
    )


def validate_epoch_times(
    record: dict[str, Any], *, expected_tasks: int, expected_epochs: int
) -> None:
    epoch_times = record.get("epoch_times")
    require(isinstance(epoch_times, list), "epoch_times no existe o no es una lista.")

    expected_count = expected_tasks * expected_epochs
    require(
        len(epoch_times) == expected_count,
        f"Numero de epoch_times incorrecto: esperado {expected_count} "
        f"({expected_tasks} tareas x {expected_epochs} epocas), encontrado {len(epoch_times)}.",
    )

    observed_pairs: list[tuple[int, int]] = []
    for index, item in enumerate(epoch_times, start=1):
        require(isinstance(item, dict), f"epoch_times[{index - 1}] no es un diccionario.")
        require(
            all(field in item for field in ("task", "epoch", "seconds")),
            f"epoch_times[{index - 1}] no contiene task, epoch y seconds: {item!r}",
        )
        try:
            task = int(item["task"])
            epoch = int(item["epoch"])
            seconds = float(item["seconds"])
        except (TypeError, ValueError) as error:
            raise AssertionError(f"epoch_times[{index - 1}] contiene valores invalidos: {item!r}") from error
        require(
            math.isfinite(seconds) and seconds >= 0,
            f"Tiempo invalido para tarea {task}, epoca {epoch}: {seconds!r}",
        )
        observed_pairs.append((task, epoch))

    expected_pairs = [
        (task, epoch)
        for task in range(1, expected_tasks + 1)
        for epoch in range(1, expected_epochs + 1)
    ]
    require(
        observed_pairs == expected_pairs,
        "La secuencia task/epoch no es completa y ordenada. "
        f"Esperada {expected_pairs}; encontrada {observed_pairs}.",
    )


def validate_accuracy_triangle(
    record: dict[str, Any], *, expected_tasks: int
) -> list[list[float | None]]:
    """Validate accuracy_i_task_j for every 1 <= i <= j <= N."""
    triangle: list[list[float | None]] = []
    for after_task in range(1, expected_tasks + 1):
        row: list[float | None] = []
        for evaluated_task in range(1, expected_tasks + 1):
            if evaluated_task > after_task:
                row.append(None)
                continue
            key = f"accuracy_{evaluated_task}_task{after_task}"
            require(
                key in record,
                f"Falta {key}: la matriz triangular de accuracies no esta completa.",
            )
            try:
                accuracy = float(record[key])
            except (TypeError, ValueError) as error:
                raise AssertionError(f"{key} no es numerica: {record[key]!r}") from error
            require(math.isfinite(accuracy), f"{key} no es finita: {accuracy!r}")
            row.append(accuracy)
        triangle.append(row)
    return triangle


def print_triangle(triangle: list[list[float | None]]) -> None:
    count = len(triangle)
    print("Matriz class-IL persistida (filas: despues de tarea j; columnas: tarea evaluada i):")
    print("despues\\eval\t" + "\t".join(f"T{i}" for i in range(1, count + 1)))
    for after_task, row in enumerate(triangle, start=1):
        values = ["-" if value is None else f"{value:.6g}" for value in row]
        print(f"T{after_task}\t\t" + "\t".join(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprueba que el smoke de Mammoth persistio args, tiempos y accuracies."
    )
    parser.add_argument("--logs", required=True, type=Path, help="Ruta al logs.pyd class-IL.")
    parser.add_argument("--model", default="l2p")
    parser.add_argument("--dataset", default="seq-cifar100-224")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--notes",
        default="gate-smoke-l2p-s0",
        help="Valor --notes que identifica el smoke. Use una cadena vacia para no filtrarlo.",
    )
    parser.add_argument("--expected-tasks", default=1, type=int)
    parser.add_argument("--expected-epochs", default=1, type=int)
    parser.add_argument("--expected-lr", default=0.0075, type=float)
    parser.add_argument("--expected-batch-size", default=64, type=int)
    parser.add_argument(
        "--after-line",
        default=0,
        type=int,
        help="Ignora registros hasta esta linea inclusive (evita aceptar un run antiguo).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require(args.expected_tasks > 0, "--expected-tasks debe ser mayor que cero.")
    require(args.expected_epochs > 0, "--expected-epochs debe ser mayor que cero.")
    require(args.expected_batch_size > 0, "--expected-batch-size debe ser mayor que cero.")
    require(args.after_line >= 0, "--after-line no puede ser negativo.")

    records = read_records(args.logs.expanduser().resolve(), after_line=args.after_line)
    notes = args.notes if args.notes else None
    line_number, record = select_record(
        records,
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        notes=notes,
        after_line=args.after_line,
    )
    validate_arguments(
        record,
        expected_lr=args.expected_lr,
        expected_batch_size=args.expected_batch_size,
    )
    validate_epoch_times(
        record,
        expected_tasks=args.expected_tasks,
        expected_epochs=args.expected_epochs,
    )
    triangle = validate_accuracy_triangle(record, expected_tasks=args.expected_tasks)

    print(f"OK: registro validado en {args.logs} (linea {line_number}).")
    print(
        "vars(args): "
        f"model={record['model']}, dataset={record['dataset']}, seed={record['seed']}, "
        f"lr_efectivo={float(record['lr']):.12g}, batch_size={record['batch_size']}, "
        f"fitting_mode={record['fitting_mode']}"
    )
    print(
        f"epoch_times: {len(record['epoch_times'])} entradas "
        f"= {args.expected_tasks} tareas x {args.expected_epochs} epocas."
    )
    print_triangle(triangle)
    print("OK: gate de persistencia superado.")


if __name__ == "__main__":
    main()
