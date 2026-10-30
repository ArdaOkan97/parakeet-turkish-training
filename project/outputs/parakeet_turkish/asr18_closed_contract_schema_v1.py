"""Only the pinned ASR18 contract-schema vocabulary; unsupported keywords stop.

This is not a general JSON Schema implementation. It validates the immutable
coverage/stage companions without adding packages to an admitted environment.
"""
import math,re

KEYS=frozenset(('$id','$schema','additionalProperties','anyOf','const','enum','items','maxItems','maximum',
    'minItems','minimum','pattern','properties','required','type'))

def require(ok,message):
    if not ok:raise ValueError(message)

def equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
    if type(a) is dict and type(b) is dict:return set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if type(a) is list and type(b) is list:return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def matches(value,kind):
    return {'object':type(value) is dict,'array':type(value) is list,'string':type(value) is str,
        'integer':type(value) is int,'number':type(value) in (int,float) and math.isfinite(value),
        'boolean':type(value) is bool,'null':value is None}.get(kind,False)

class ClosedSchema:
    def __init__(self,schema):
        require(type(schema) is dict and set(schema)<=KEYS,'Unsupported contract schema vocabulary; stop rather than ignore')
        self.schema=schema
        kinds=schema.get('type',[]);kinds=[kinds] if type(kinds) is str else kinds
        require(type(kinds) is list and all(v in ('object','array','string','integer','number','boolean','null') for v in kinds),'Unknown contract schema type')
        require(schema.get('additionalProperties',False) is False,'Only explicit closed object schemas supported')
        for value in schema.get('properties',{}).values():ClosedSchema(value)
        if 'items' in schema:ClosedSchema(schema['items'])
        for value in schema.get('anyOf',[]):ClosedSchema(value)

    def validate(self,value):
        s=self.schema
        if 'anyOf' in s:
            success=False
            for candidate in s['anyOf']:
                try:ClosedSchema(candidate).validate(value);success=True;break
                except ValueError:pass
            require(success,'No allowed contract schema branch matches')
        if 'const' in s:require(equal(value,s['const']),'Contract constant differs')
        if 'enum' in s:require(any(equal(value,v) for v in s['enum']),'Contract enum differs')
        if 'type' in s:
            kinds=[s['type']] if type(s['type']) is str else s['type']
            require(any(matches(value,v) for v in kinds),'Contract value type differs')
        if type(value) is dict:
            props=s.get('properties',{});require(set(s.get('required',[]))<=set(value),'Required contract keys missing')
            if s.get('additionalProperties') is False:require(set(value)<=set(props),'Additional contract keys prohibited')
            for key,item in value.items():
                if key in props:ClosedSchema(props[key]).validate(item)
        if type(value) is list:
            require(s.get('minItems',0)<=len(value)<=s.get('maxItems',len(value)),'Contract array length differs')
            if 'items' in s:
                for item in value:ClosedSchema(s['items']).validate(item)
        if type(value) in (int,float):
            require(math.isfinite(value) and s.get('minimum',value)<=value<=s.get('maximum',value),'Contract numeric bound differs')
        if 'pattern' in s:require(type(value) is str and re.search(s['pattern'],value) is not None,'Contract string pattern differs')
