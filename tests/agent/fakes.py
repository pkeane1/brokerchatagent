from types import SimpleNamespace as NS

import anthropic


def usage(input_tokens=100, output_tokens=20):
    return NS(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0,
              cache_creation_input_tokens=None)


def response(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content), usage=usage())


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeConnectionError(anthropic.APIConnectionError):
    def __init__(self):  # skip the SDK constructor, which needs a real HTTP request object
        Exception.__init__(self, "connection error")


class FakeClient:
    """Stands in for anthropic.Anthropic: returns (or raises) scripted items in order."""

    def __init__(self, *items):
        self._items = list(items)
        self.calls = []
        self.beta = NS(messages=self)

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
