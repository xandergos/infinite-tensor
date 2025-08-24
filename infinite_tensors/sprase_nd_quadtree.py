from typing import Dict, Tuple, Sequence, Optional

class _Node:
	__slots__ = ("count", "children")
	def __init__(self) -> None:
		self.count: int = 0
		self.children: Dict[int, "_Node"] = {}

class SparseNDQuadtree:
	def __init__(self, dims: int) -> None:
		if dims <= 0:
			raise ValueError("dims must be >= 1")
		self.d = dims
		self.root = _Node()
		self.root_min = [0] * self.d  # inclusive lower corner
		self.root_size = 1            # side length as power-of-two (#cells per dim)
		self._points: set[Tuple[int, ...]] = set()

	def set_one(self, p: Sequence[int]) -> bool:
		"""Set cell at p to 1. Returns True if changed."""
		pt = self._as_point(p)
		if pt in self._points:
			return False
		self._expand_to_include(pt)
		self._insert(self.root, self.root_min, self.root_size, pt)
		self._points.add(pt)
		return True

	def clear(self, p: Sequence[int]) -> bool:
		"""Set cell at p to 0. Returns True if changed."""
		pt = self._as_point(p)
		if pt not in self._points:
			return False
		self._remove(self.root, self.root_min, self.root_size, pt)
		self._points.remove(pt)
		self._maybe_shrink_root()
		return True

	def has_zero(self, lo: Sequence[int], hi: Sequence[int]) -> bool:
		"""Closed box [lo, hi], per-dimension. True if any 0 exists inside."""
		lo_t, hi_t = self._as_box(lo, hi)
		vol = self._volume(lo_t, hi_t)
		if vol == 0:
			return False
		count = self.count_ones(lo_t, hi_t)
		return count < vol

	def count_ones(self, lo: Sequence[int], hi: Sequence[int]) -> int:
		lo_t, hi_t = self._as_box(lo, hi)
		if self.root.count == 0:
			return 0
		return self._range_count(self.root, self.root_min, self.root_size, lo_t, hi_t)

	# ---------------- internal helpers ----------------

	def _as_point(self, p: Sequence[int]) -> Tuple[int, ...]:
		if len(p) != self.d:
			raise ValueError("point dimensionality mismatch")
		return tuple(int(x) for x in p)

	def _as_box(self, lo: Sequence[int], hi: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
		if len(lo) != self.d or len(hi) != self.d:
			raise ValueError("box dimensionality mismatch")
		lo_t = tuple(int(x) for x in lo)
		hi_t = tuple(int(x) for x in hi)
		# Normalize: ensure lo <= hi per dim
		lo_n = tuple(min(a, b) for a, b in zip(lo_t, hi_t))
		hi_n = tuple(max(a, b) for a, b in zip(lo_t, hi_t))
		return lo_n, hi_n

	def _volume(self, lo: Tuple[int, ...], hi: Tuple[int, ...]) -> int:
		vol = 1
		for a, b in zip(lo, hi):
			side = b - a + 1
			if side <= 0:
				return 0
			vol *= side
		return vol

	def _contains_point(self, mn: Sequence[int], size: int, p: Tuple[int, ...]) -> bool:
		for i in range(self.d):
			if p[i] < mn[i] or p[i] >= mn[i] + size:
				return False
		return True

	def _expand_to_include(self, p: Tuple[int, ...]) -> None:
		while not self._contains_point(self.root_min, self.root_size, p):
			new_min = list(self.root_min)
			for i in range(self.d):
				if p[i] < self.root_min[i]:
					new_min[i] = self.root_min[i] - self.root_size
			new_root = _Node()
			new_root.count = self.root.count
			child_index = 0
			half = self.root_size
			for i in range(self.d):
				if new_min[i] < self.root_min[i]:
					child_index |= (1 << i)  # old root sits in upper half along this dim
			new_root.children[child_index] = self.root
			self.root = new_root
			self.root_min = new_min
			self.root_size = self.root_size * 2

	def _insert(self, node: _Node, mn: Sequence[int], size: int, p: Tuple[int, ...]) -> None:
		node.count += 1
		if size == 1:
			return
		half = size // 2
		mid = [mn[i] + half for i in range(self.d)]
		idx = 0
		child_min = [0] * self.d
		for i in range(self.d):
			if p[i] >= mid[i]:
				idx |= (1 << i)
				child_min[i] = mid[i]
			else:
				child_min[i] = mn[i]
		ch = node.children.get(idx)
		if ch is None:
			ch = _Node()
			node.children[idx] = ch
		self._insert(ch, child_min, half, p)

	def _remove(self, node: _Node, mn: Sequence[int], size: int, p: Tuple[int, ...]) -> bool:
		# returns True if node becomes empty
		node.count -= 1
		if size == 1:
			return node.count == 0
		half = size // 2
		mid = [mn[i] + half for i in range(self.d)]
		idx = 0
		child_min = [0] * self.d
		for i in range(self.d):
			if p[i] >= mid[i]:
				idx |= (1 << i)
				child_min[i] = mid[i]
			else:
				child_min[i] = mn[i]
		ch = node.children.get(idx)
		if ch is not None:
			empty = self._remove(ch, child_min, half, p)
			if empty:
				del node.children[idx]
		return node.count == 0

	def _maybe_shrink_root(self) -> None:
		while self.root_size > 1 and self.root.count > 0 and len(self.root.children) == 1:
			(idx, child) = next(iter(self.root.children.items()))
			half = self.root_size // 2
			for i in range(self.d):
				if (idx >> i) & 1:
					self.root_min[i] += half
			self.root = child
			self.root_size = half

	def _node_box_contains(self, mn: Sequence[int], size: int, lo: Tuple[int, ...], hi: Tuple[int, ...]) -> bool:
		for i in range(self.d):
			a = mn[i]
			b = a + size - 1
			if a < lo[i] or b > hi[i]:
				return False
		return True

	def _node_box_intersects(self, mn: Sequence[int], size: int, lo: Tuple[int, ...], hi: Tuple[int, ...]) -> bool:
		for i in range(self.d):
			a0 = mn[i]
			a1 = a0 + size - 1
			if a0 > hi[i] or a1 < lo[i]:
				return False
		return True

	def _range_count(self, node: _Node, mn: Sequence[int], size: int, lo: Tuple[int, ...], hi: Tuple[int, ...]) -> int:
		if node.count == 0:
			return 0
		if self._node_box_contains(mn, size, lo, hi):
			return node.count
		if not self._node_box_intersects(mn, size, lo, hi):
			return 0
		if size == 1:
			return 1  # node.count > 0 implies exactly one 1 here
		total = 0
		half = size // 2
		for idx, ch in node.children.items():
			child_min = [0] * self.d
			for i in range(self.d):
				if (idx >> i) & 1:
					child_min[i] = mn[i] + half
				else:
					child_min[i] = mn[i]
			total += self._range_count(ch, child_min, half, lo, hi)
		return total
