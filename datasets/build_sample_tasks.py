"""Builds datasets/sample_tasks.json: 30 small custom Python tasks, 6 per workflow.

Written for this project (no external benchmark data, so no licence questions).
They exist to test the pipeline and for a preliminary demonstration; 30 samples are
far too few for research conclusions.

Fields used in the prompt: task, context. Everything else (reference_answer,
expected_behavior, tests, reference_solution, buggy_solutions, expected_answers,
keywords) is hidden from the model and used only for evaluation.

Run:  python datasets/build_sample_tasks.py
"""
import hashlib
import json
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sample_tasks.json"


def d(s):
    return textwrap.dedent(s).strip("\n") + "\n"


TASKS = []


def add(workflow, group, task, context, eval_type, reference_answer, expected_behavior, **kw):
    TASKS.append(dict(workflow=workflow, group_id=group, language="python", task=task, context=d(context),
                      eval_type=eval_type, reference_answer=reference_answer,
                      expected_behavior=expected_behavior, max_new_tokens=256, **kw))


CODE_ONLY = " Return only the complete Python code in one ```python block."

# ---------------------------------------------------------------- completion (6)
add("completion", "g_binary_search", "Complete the function so it satisfies its docstring." + CODE_ONLY, '''
    def binary_search(items, target):
        """Return the index of target in the sorted list items, or -1 if absent.
        Must run in O(log n)."""
    ''', "unit_tests", "Iterative binary search returning index or -1.", "Index of target or -1.",
    entry_point="binary_search",
    reference_solution=d('''
    def binary_search(items, target):
        lo, hi = 0, len(items) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if items[mid] == target:
                return mid
            if items[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1
    '''),
    tests=["assert binary_search([1, 3, 5, 7, 9], 7) == 3", "assert binary_search([1, 3, 5, 7, 9], 1) == 0",
           "assert binary_search([1, 3, 5, 7, 9], 9) == 4", "assert binary_search([1, 3, 5], 4) == -1",
           "assert binary_search([], 1) == -1"])

add("completion", "g_word_freq", "Complete the function so it satisfies its docstring." + CODE_ONLY, '''
    def top_k_words(text, k):
        """Return the k most frequent words in text as a list of (word, count) tuples.
        Words are lower-cased and split on whitespace; punctuation .,!? is stripped
        from both ends of each word. Ties are broken alphabetically."""
    ''', "unit_tests", "Counts words and sorts by (-count, word).", "List of (word, count).",
    entry_point="top_k_words",
    reference_solution=d('''
    from collections import Counter
    def top_k_words(text, k):
        words = [w.strip(".,!?").lower() for w in text.split()]
        counts = Counter(w for w in words if w)
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    '''),
    tests=["assert top_k_words('a b a c b a', 2) == [('a', 3), ('b', 2)]",
           "assert top_k_words('Hello, hello world!', 1) == [('hello', 2)]",
           "assert top_k_words('x y z', 2) == [('x', 1), ('y', 1)]",
           "assert top_k_words('', 3) == []"])

add("completion", "g_flatten", "Complete the function so it satisfies its docstring." + CODE_ONLY, '''
    def flatten(nested):
        """Flatten arbitrarily nested lists and tuples into a single list, keeping order.
        Strings are treated as atoms, not iterated."""
    ''', "unit_tests", "Recursive flatten of lists/tuples.", "Flat list.",
    entry_point="flatten",
    reference_solution=d('''
    def flatten(nested):
        out = []
        for x in nested:
            if isinstance(x, (list, tuple)):
                out.extend(flatten(x))
            else:
                out.append(x)
        return out
    '''),
    tests=["assert flatten([1, [2, [3, (4, 5)]], 6]) == [1, 2, 3, 4, 5, 6]", "assert flatten([]) == []",
           "assert flatten(['ab', ['cd']]) == ['ab', 'cd']", "assert flatten([[[]]]) == []"])

add("completion", "g_lru", "Complete the class so it satisfies its docstring." + CODE_ONLY, '''
    class LRUCache:
        """Least-recently-used cache with a fixed capacity.
        get(key) returns the value or -1 and marks the key as recently used.
        put(key, value) inserts/updates and evicts the least recently used key when full."""

        def __init__(self, capacity):
            ...

        def get(self, key):
            ...

        def put(self, key, value):
            ...
    ''', "unit_tests", "OrderedDict-based LRU cache.", "LRU eviction semantics.",
    entry_point="LRUCache",
    reference_solution=d('''
    from collections import OrderedDict
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.data = OrderedDict()
        def get(self, key):
            if key not in self.data:
                return -1
            self.data.move_to_end(key)
            return self.data[key]
        def put(self, key, value):
            if key in self.data:
                self.data.move_to_end(key)
            self.data[key] = value
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)
    '''),
    tests=["c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1",
           "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3); assert c.get(2) == -1",
           "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3); assert c.get(1) == 1 and c.get(3) == 3",
           "c = LRUCache(1); c.put(1, 1); c.put(1, 5); assert c.get(1) == 5",
           "c = LRUCache(1); assert c.get(9) == -1"])

add("completion", "g_intervals", "Complete the function so it satisfies its docstring." + CODE_ONLY, '''
    def merge_intervals(intervals):
        """Merge overlapping [start, end] intervals (touching intervals also merge).
        Input may be unsorted. Return a new sorted list of merged intervals."""
    ''', "unit_tests", "Sort by start then merge.", "Sorted merged intervals.",
    entry_point="merge_intervals",
    reference_solution=d('''
    def merge_intervals(intervals):
        out = []
        for s, e in sorted(intervals):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    '''),
    tests=["assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
           "assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]",
           "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]", "assert merge_intervals([]) == []",
           "assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]"])

add("completion", "g_roman", "Complete the function so it satisfies its docstring." + CODE_ONLY, '''
    def to_roman(n):
        """Convert an integer 1 <= n <= 3999 to a Roman numeral string.
        Raise ValueError outside that range."""
    ''', "unit_tests", "Greedy value/symbol table.", "Roman numeral string.",
    entry_point="to_roman",
    reference_solution=d('''
    def to_roman(n):
        if not 1 <= n <= 3999:
            raise ValueError(n)
        table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
                 (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
        out = ''
        for v, s in table:
            while n >= v:
                out += s
                n -= v
        return out
    '''),
    tests=["assert to_roman(1994) == 'MCMXCIV'", "assert to_roman(4) == 'IV'", "assert to_roman(3999) == 'MMMCMXCIX'",
           "try:\n    to_roman(0)\n    raise AssertionError('no error')\nexcept ValueError:\n    pass"])

# ---------------------------------------------------------------- debugging (6)
DBG = "The function below has a bug. Find it and return the corrected function." + CODE_ONLY

add("debugging", "g_binary_search", DBG, '''
    def binary_search(items, target):
        """Return the index of target in the sorted list items, or -1 if absent."""
        lo, hi = 0, len(items) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if items[mid] == target:
                return mid
            if items[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1
    ''', "unit_tests", "Loop condition must be lo <= hi; otherwise single-element ranges are skipped.",
    "Finds the last element and single-element lists.", entry_point="binary_search",
    reference_solution=TASKS[0]["reference_solution"], tests=TASKS[0]["tests"] + ["assert binary_search([4], 4) == 0"])

add("debugging", "g_avg", DBG, '''
    def moving_average(values, window):
        """Return the list of averages of each consecutive window of size `window`."""
        out = []
        for i in range(len(values) - window):
            out.append(sum(values[i:i + window]) / window)
        return out
    ''', "unit_tests", "Off-by-one: range must be len(values) - window + 1.", "Includes the final window.",
    entry_point="moving_average",
    reference_solution=d('''
    def moving_average(values, window):
        out = []
        for i in range(len(values) - window + 1):
            out.append(sum(values[i:i + window]) / window)
        return out
    '''),
    tests=["assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]", "assert moving_average([5], 1) == [5.0]",
           "assert moving_average([1, 2], 3) == []"])

add("debugging", "g_mutable_default", DBG, '''
    def add_tag(tag, tags=[]):
        """Return a new list made of `tags` plus `tag`. Calls must not affect each other."""
        tags.append(tag)
        return tags
    ''', "unit_tests", "Mutable default argument is shared between calls; use None.", "Independent results per call.",
    entry_point="add_tag",
    reference_solution=d('''
    def add_tag(tag, tags=None):
        tags = list(tags) if tags else []
        tags.append(tag)
        return tags
    '''),
    tests=["add_tag('a'); assert add_tag('b') == ['b']", "assert add_tag('x', ['y']) == ['y', 'x']",
           "base = ['k']; add_tag('z', base); assert base == ['k']"])

add("debugging", "g_palindrome", DBG, '''
    def is_palindrome(s):
        """True if s reads the same forwards and backwards, ignoring case and
        non-alphanumeric characters."""
        cleaned = [c for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]
    ''', "unit_tests", "Case is not normalised; lower-case the characters.", "Case-insensitive check.",
    entry_point="is_palindrome",
    reference_solution=d('''
    def is_palindrome(s):
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]
    '''),
    tests=["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('race a car')",
           "assert is_palindrome('')", "assert is_palindrome('Aa')"])

add("debugging", "g_dict_iter", DBG, '''
    def remove_small(scores, threshold):
        """Remove, in place, every entry of the dict `scores` whose value is below
        threshold, and return the dict."""
        for name in scores:
            if scores[name] < threshold:
                del scores[name]
        return scores
    ''', "unit_tests", "Deleting while iterating raises RuntimeError; iterate over a copy of the keys.",
    "Removes small entries without error.", entry_point="remove_small",
    reference_solution=d('''
    def remove_small(scores, threshold):
        for name in list(scores):
            if scores[name] < threshold:
                del scores[name]
        return scores
    '''),
    tests=["assert remove_small({'a': 1, 'b': 5, 'c': 2}, 3) == {'b': 5}",
           "d = {'a': 1}; remove_small(d, 2); assert d == {}", "assert remove_small({}, 1) == {}"])

add("debugging", "g_fib", DBG, '''
    def fib(n):
        """Return the n-th Fibonacci number with fib(0) == 0 and fib(1) == 1."""
        a, b = 0, 1
        for _ in range(n):
            a = b
            b = a + b
        return a
    ''', "unit_tests", "Sequential assignment loses the old value of a; use tuple assignment a, b = b, a + b.",
    "Correct Fibonacci numbers.", entry_point="fib",
    reference_solution=d('''
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    '''),
    tests=["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(2) == 1", "assert fib(10) == 55"])

# ---------------------------------------------------------------- testgen (6)
TG = ("Write pytest unit tests for the function below. Import it with `from solution import {fn}`. "
      "Cover normal cases and edge cases. Return only the test code in one ```python block.")

add("testgen", "g_clamp", TG.format(fn="clamp"), '''
    def clamp(x, low, high):
        """Limit x to the closed range [low, high]. Raise ValueError if low > high."""
        if low > high:
            raise ValueError("low > high")
        return max(low, min(x, high))
    ''', "testgen", "Tests for below, inside, above range and invalid bounds.", "Tests pass on correct code, fail on bugs.",
    module_name="solution",
    reference_solution=d('''
    def clamp(x, low, high):
        if low > high:
            raise ValueError("low > high")
        return max(low, min(x, high))
    '''),
    buggy_solutions=[d('''
    def clamp(x, low, high):
        return max(low, min(x, high))
    '''), d('''
    def clamp(x, low, high):
        if low > high:
            raise ValueError("low > high")
        return min(low, max(x, high))
    ''')])

add("testgen", "g_slug", TG.format(fn="slugify"), '''
    import re

    def slugify(title):
        """Lower-case title, replace runs of non-alphanumeric characters with a single
        '-', and strip leading/trailing '-'."""
        return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    ''', "testgen", "Tests for spaces, punctuation runs, edges and case.", "Tests pass on correct code, fail on bugs.",
    module_name="solution",
    reference_solution=d('''
    import re
    def slugify(title):
        return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    '''),
    buggy_solutions=[d('''
    import re
    def slugify(title):
        return re.sub(r"[^a-z0-9]+", "-", title.lower())
    '''), d('''
    import re
    def slugify(title):
        return re.sub(r"[^a-z0-9]", "-", title.lower()).strip("-")
    ''')])

add("testgen", "g_chunks", TG.format(fn="chunk"), '''
    def chunk(items, size):
        """Split items into consecutive lists of length `size`; the last may be shorter.
        Raise ValueError if size < 1."""
        if size < 1:
            raise ValueError("size must be >= 1")
        return [items[i:i + size] for i in range(0, len(items), size)]
    ''', "testgen", "Tests for exact split, remainder, empty input and invalid size.", "Tests pass on correct code, fail on bugs.",
    module_name="solution",
    reference_solution=d('''
    def chunk(items, size):
        if size < 1:
            raise ValueError("size must be >= 1")
        return [items[i:i + size] for i in range(0, len(items), size)]
    '''),
    buggy_solutions=[d('''
    def chunk(items, size):
        if size < 1:
            raise ValueError("size must be >= 1")
        return [items[i:i + size] for i in range(0, len(items) - size + 1, size)]
    '''), d('''
    def chunk(items, size):
        return [items[i:i + size] for i in range(0, len(items), max(size, 1))]
    ''')])

add("testgen", "g_roman", TG.format(fn="to_roman"), TASKS[5]["reference_solution"],
    "testgen", "Tests for subtractive forms, bounds and ValueError.", "Tests pass on correct code, fail on bugs.",
    module_name="solution", reference_solution=TASKS[5]["reference_solution"],
    buggy_solutions=[TASKS[5]["reference_solution"].replace("(900, 'CM'), ", "").replace("(400, 'CD'), ", "")
                     .replace("(90, 'XC'), ", "").replace("(40, 'XL'), ", "").replace("(9, 'IX'), ", "")
                     .replace("(4, 'IV'), ", ""),
                     TASKS[5]["reference_solution"].replace("if not 1 <= n <= 3999:\n        raise ValueError(n)\n", "")])

add("testgen", "g_parse_kv", TG.format(fn="parse_kv"), '''
    def parse_kv(line):
        """Parse 'k1=v1;k2=v2' into a dict. Whitespace around keys and values is
        stripped, empty segments are ignored, and a segment without '=' raises ValueError."""
        out = {}
        for part in line.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(part)
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    ''', "testgen", "Tests for normal parsing, whitespace, empty segments, '=' in value, invalid segment.",
    "Tests pass on correct code, fail on bugs.", module_name="solution",
    reference_solution=d('''
    def parse_kv(line):
        out = {}
        for part in line.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(part)
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    '''),
    buggy_solutions=[d('''
    def parse_kv(line):
        out = {}
        for part in line.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(part)
            k, v = part.split("=", 1)
            out[k] = v
        return out
    '''), d('''
    def parse_kv(line):
        out = {}
        for part in line.split(";"):
            if not part.strip() or "=" not in part:
                continue
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    ''')])

add("testgen", "g_median", TG.format(fn="median"), '''
    def median(values):
        """Return the median of a non-empty list of numbers (mean of the two middle
        values for even length). Raise ValueError on an empty list. Must not modify the input."""
        if not values:
            raise ValueError("empty")
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
    ''', "testgen", "Tests for odd, even, unsorted input, empty list.", "Tests pass on correct code, fail on bugs.",
    module_name="solution",
    reference_solution=d('''
    def median(values):
        if not values:
            raise ValueError("empty")
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
    '''),
    buggy_solutions=[d('''
    def median(values):
        if not values:
            raise ValueError("empty")
        s = sorted(values)
        return s[len(s) // 2]
    '''), d('''
    def median(values):
        if not values:
            raise ValueError("empty")
        m = len(values) // 2
        return values[m] if len(values) % 2 else (values[m - 1] + values[m]) / 2
    ''')])

# ---------------------------------------------------------------- longcode (6)
INVENTORY = '''
    from dataclasses import dataclass, field

    TAX_RATE = 0.08
    BULK_THRESHOLD = 10
    BULK_DISCOUNT = 0.05


    @dataclass
    class Item:
        sku: str
        price: float
        quantity: int = 0


    @dataclass
    class Inventory:
        items: dict = field(default_factory=dict)

        def add(self, sku, price, quantity):
            if sku in self.items:
                self.items[sku].quantity += quantity
            else:
                self.items[sku] = Item(sku, price, quantity)

        def remove(self, sku, quantity):
            item = self.items.get(sku)
            if item is None or item.quantity < quantity:
                raise KeyError(f"not enough {sku}")
            item.quantity -= quantity
            if item.quantity == 0:
                del self.items[sku]

        def value(self):
            return sum(i.price * i.quantity for i in self.items.values())


    def order_total(inventory, sku, quantity):
        """Price of an order including bulk discount and tax."""
        item = inventory.items[sku]
        subtotal = item.price * quantity
        if quantity >= BULK_THRESHOLD:
            subtotal *= 1 - BULK_DISCOUNT
        return round(subtotal * (1 + TAX_RATE), 2)
'''

add("longcode", "g_inventory", "Explain what `order_total` computes, step by step, including when a discount applies.",
    INVENTORY, "keywords",
    "order_total multiplies the item's price by the quantity, applies a 5% bulk discount when the quantity is at least 10, then adds 8% tax and rounds to 2 decimals.",
    "Mentions price x quantity, bulk discount at 10+, 5%, tax 8%, rounding.",
    keywords=[["price"], ["quantity"], ["discount"], ["10", "ten", "bulk_threshold"], ["tax", "tax_rate"], ["round", "rounds", "rounded", "rounding"]])

add("longcode", "g_inventory", "What does `order_total(inv, 'A', 10)` return if item 'A' has price 10.0? Give the number.",
    INVENTORY, "answer_match", "102.6", "10*10=100, -5% = 95, +8% tax = 102.6.", expected_answers=["102.6", "102.60"])

TOKENIZER = '''
    import re

    TOKEN_SPEC = [
        ("NUMBER", r"\\d+(?:\\.\\d+)?"),
        ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("OP", r"[+\\-*/()]"),
        ("SKIP", r"[ \\t]+"),
        ("MISMATCH", r"."),
    ]
    MASTER = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))


    def tokenize(text):
        for m in MASTER.finditer(text):
            kind, value = m.lastgroup, m.group()
            if kind == "SKIP":
                continue
            if kind == "MISMATCH":
                raise SyntaxError(f"unexpected {value!r}")
            yield kind, value


    class Parser:
        def __init__(self, text, env=None):
            self.tokens = list(tokenize(text))
            self.pos = 0
            self.env = env or {}

        def peek(self):
            return self.tokens[self.pos][1] if self.pos < len(self.tokens) else None

        def take(self):
            tok = self.tokens[self.pos]
            self.pos += 1
            return tok

        def expr(self):
            value = self.term()
            while self.peek() in ("+", "-"):
                op = self.take()[1]
                value = value + self.term() if op == "+" else value - self.term()
            return value

        def term(self):
            value = self.factor()
            while self.peek() in ("*", "/"):
                op = self.take()[1]
                value = value * self.factor() if op == "*" else value / self.factor()
            return value

        def factor(self):
            kind, value = self.take()
            if kind == "NUMBER":
                return float(value)
            if kind == "IDENT":
                return float(self.env[value])
            if value == "(":
                inner = self.expr()
                self.take()
                return inner
            if value == "-":
                return -self.factor()
            raise SyntaxError(f"unexpected {value!r}")


    def evaluate(text, **env):
        return Parser(text, env).expr()
'''

add("longcode", "g_parser", "Explain how this module evaluates an arithmetic expression and how operator precedence is handled.",
    TOKENIZER, "keywords",
    "The text is tokenized with a regular expression, then a recursive-descent parser evaluates it: expr handles + and -, term handles * and / so multiplication binds tighter, factor handles numbers, variables from env, parentheses and unary minus.",
    "Mentions tokenizer/regex, recursive descent, expr/term/factor precedence, parentheses, variables.",
    keywords=[["token", "tokens", "tokenize", "tokenizer", "tokenizes"], ["regex", "regular expression", "re"],
              ["recursive", "recursive descent", "recursion"], ["precedence", "binds", "higher priority"],
              ["parentheses", "parenthesis", "brackets"], ["env", "variable", "variables", "environment"]])

add("longcode", "g_parser", "What value does `evaluate('2 + 3 * (x - 1)', x=4)` return? Give the number.",
    TOKENIZER, "answer_match", "11.0", "3*(4-1)=9, 2+9=11.", expected_answers=["11.0", "11"])

RATE_LIMITER = '''
    import time


    class TokenBucket:
        """Allow `rate` requests per second with bursts up to `capacity`."""

        def __init__(self, rate, capacity, clock=time.monotonic):
            self.rate = rate
            self.capacity = capacity
            self.tokens = capacity
            self.clock = clock
            self.last = clock()

        def _refill(self):
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now

        def allow(self, cost=1):
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


    class KeyedLimiter:
        def __init__(self, rate, capacity, clock=time.monotonic):
            self.buckets = {}
            self.rate, self.capacity, self.clock = rate, capacity, clock

        def allow(self, key, cost=1):
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(self.rate, self.capacity, self.clock)
            return self.buckets[key].allow(cost)
'''

add("longcode", "g_ratelimit", "Explain what this code does and how `allow` decides whether a request is allowed.",
    RATE_LIMITER, "keywords",
    "It implements a token-bucket rate limiter: tokens refill continuously at `rate` per second up to `capacity`; allow refills, then permits the request and subtracts its cost if enough tokens remain. KeyedLimiter keeps one bucket per key.",
    "Mentions token bucket, refill rate, capacity/burst, cost subtraction, per-key buckets.",
    keywords=[["token bucket", "bucket"], ["refill", "refills", "replenish"], ["rate"], ["capacity", "burst"],
              ["cost"], ["key", "per-key", "per key", "each key"]])

add("longcode", "g_ratelimit",
    "With a fake clock starting at 0, `b = TokenBucket(rate=1, capacity=2, clock=clock)`. At time 0, `b.allow()` "
    "is called three times. How many of the three calls return True? Answer with a single number.",
    RATE_LIMITER, "answer_match", "2", "Capacity 2, no time passes, so the third call is denied.", expected_answers=["2", "two"])

# ---------------------------------------------------------------- repoqa (6)
REPO_A = '''
    # file: shop/config.py
    DEFAULT_CURRENCY = "EUR"
    FREE_SHIPPING_OVER = 50.0
    SHIPPING_FEE = 4.99


    # file: shop/models.py
    from dataclasses import dataclass


    @dataclass
    class LineItem:
        name: str
        unit_price: float
        qty: int


    @dataclass
    class Cart:
        items: list
        coupon: str = ""


    # file: shop/pricing.py
    from shop.config import FREE_SHIPPING_OVER, SHIPPING_FEE
    from shop.coupons import coupon_discount


    def subtotal(cart):
        return sum(i.unit_price * i.qty for i in cart.items)


    def shipping(amount):
        return 0.0 if amount >= FREE_SHIPPING_OVER else SHIPPING_FEE


    def cart_total(cart):
        base = subtotal(cart)
        discounted = base - coupon_discount(cart.coupon, base)
        return round(discounted + shipping(discounted), 2)


    # file: shop/coupons.py
    COUPONS = {"WELCOME10": 0.10, "VIP20": 0.20}


    def coupon_discount(code, amount):
        rate = COUPONS.get(code.upper(), 0.0)
        return amount * rate


    # file: shop/api.py
    from shop.models import Cart, LineItem
    from shop.pricing import cart_total


    def checkout(payload):
        items = [LineItem(**i) for i in payload["items"]]
        cart = Cart(items, payload.get("coupon", ""))
        return {"total": cart_total(cart), "currency": payload.get("currency", "EUR")}
'''

add("repoqa", "g_shop", "In this repository, which file defines the function that decides the shipping fee? Answer with the file path.",
    REPO_A, "answer_match", "shop/pricing.py", "The shipping() function lives in shop/pricing.py.",
    expected_answers=["shop/pricing.py"])
add("repoqa", "g_shop", "Which function in which file applies the coupon percentage? Answer as file_path:function_name.",
    REPO_A, "answer_match", "shop/coupons.py:coupon_discount", "coupon_discount in shop/coupons.py.",
    expected_answers=["coupon_discount"])
add("repoqa", "g_shop",
    "What total does `checkout` return for one item with unit_price 30 and qty 2 and coupon 'welcome10'? Give the number.",
    REPO_A, "answer_match", "54.0", "60 - 10% = 54 >= 50 so free shipping; total 54.0.", expected_answers=["54.0", "54"])

REPO_B = '''
    # file: jobs/registry.py
    _HANDLERS = {}


    def register(name):
        def deco(fn):
            _HANDLERS[name] = fn
            return fn
        return deco


    def get_handler(name):
        return _HANDLERS[name]


    # file: jobs/retry.py
    import time

    MAX_ATTEMPTS = 3
    BACKOFF_SECONDS = 0.5


    def with_retry(fn, *args, sleep=time.sleep):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return fn(*args)
            except ConnectionError:
                if attempt == MAX_ATTEMPTS:
                    raise
                sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))


    # file: jobs/handlers.py
    from jobs.registry import register


    @register("resize")
    def resize_image(path, width):
        return f"{path}@{width}"


    @register("email")
    def send_email(address, body):
        return f"sent to {address}"


    # file: jobs/worker.py
    from jobs.registry import get_handler
    from jobs.retry import with_retry
    import jobs.handlers  # noqa: F401  (registers handlers)


    def run_job(job):
        handler = get_handler(job["type"])
        return with_retry(handler, *job["args"])
'''

add("repoqa", "g_jobs", "How does `run_job` find the function to call for a job type? Name the mechanism and the file where handlers are registered.",
    REPO_B, "keywords",
    "Handlers register themselves with the @register decorator into the _HANDLERS dict in jobs/registry.py; jobs/handlers.py is imported by jobs/worker.py so the decorators run, and run_job looks the handler up with get_handler.",
    "Mentions decorator registry, _HANDLERS dict, jobs/handlers.py import, get_handler.",
    keywords=[["register", "decorator"], ["_handlers", "dict", "dictionary", "registry"], ["jobs/handlers.py", "handlers.py"],
              ["get_handler"], ["import", "imported", "imports"]])
add("repoqa", "g_jobs", "How many times in total will a handler be called if it always raises ConnectionError? Answer with a single number.",
    REPO_B, "answer_match", "3", "MAX_ATTEMPTS = 3, then the error is re-raised.", expected_answers=["3", "three"])
add("repoqa", "g_jobs", "What is the total time slept by `with_retry` before it finally re-raises, if every attempt fails? Give the number of seconds.",
    REPO_B, "answer_match", "1.5", "Sleeps 0.5 then 1.0 seconds = 1.5.", expected_answers=["1.5"])


def split_for(group_id):
    """Deterministic group-level split (60/20/20) so variants of one task never straddle splits."""
    h = int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else ("validation" if h < 8 else "test")


def main():
    counters = {}
    rows = []
    for i, t in enumerate(TASKS, 1):
        counters[t["workflow"]] = counters.get(t["workflow"], 0) + 1
        rows.append({"sample_id": f"python_{i:03d}", "split": split_for(t["group_id"]), **t})
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} tasks to {OUT}: {counters}")
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print("split sizes:", splits)


if __name__ == "__main__":
    main()
