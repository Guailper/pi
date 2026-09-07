from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    pass


def validate(schema: dict[str, Any], value: Any, path: str = '$') -> Any:
    expected = schema.get('type')
    if expected == 'object':
        if not isinstance(value, dict):
            raise SchemaError(f'{path}: expected object')
        for key in schema.get('required', []):
            if key not in value:
                raise SchemaError(f'{path}.{key}: required field missing')
        props = schema.get('properties', {})
        for key, child in props.items():
            if key in value:
                validate(child, value[key], f'{path}.{key}')
    elif expected == 'array':
        if not isinstance(value, list):
            raise SchemaError(f'{path}: expected array')
        item_schema = schema.get('items')
        if item_schema:
            for idx, item in enumerate(value):
                validate(item_schema, item, f'{path}[{idx}]')
    elif expected == 'string' and not isinstance(value, str):
        raise SchemaError(f'{path}: expected string')
    elif expected == 'integer' and (not isinstance(value, int) or isinstance(value, bool)):
        raise SchemaError(f'{path}: expected integer')
    elif expected == 'number' and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise SchemaError(f'{path}: expected number')
    elif expected == 'boolean' and not isinstance(value, bool):
        raise SchemaError(f'{path}: expected boolean')

    if 'enum' in schema and value not in schema['enum']:
        raise SchemaError(f'{path}: expected one of {schema["enum"]!r}')
    return value
