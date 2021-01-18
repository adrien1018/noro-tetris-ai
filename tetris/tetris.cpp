#define PY_SSIZE_T_CLEAN
#include <cstdint>
#include <array>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>
#if __has_include(<python3.9/Python.h>)
#include <python3.9/Python.h>
#else
#include <Python.h>
#endif

using Entry = int32_t;
using Poly = std::array<std::pair<int, int>, 4>;
const int kN = 21, kM = 10;
const std::vector<Poly> kBlocks[] = {
    {{{{0, -1}, {0, 0}, {0, 1}, {-1, 0}}}, // T
     {{{1, 0}, {0, 0}, {0, 1}, {-1, 0}}},
     {{{1, 0}, {0, 0}, {0, 1}, {0, -1}}},
     {{{1, 0}, {0, 0}, {-1, 0}, {0, -1}}}},
    {{{{-1, -1}, {0, -1}, {0, 0}, {0, 1}}}, // J
     {{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}}},
     {{{0, -1}, {0, 0}, {0, 1}, {1, 1}}},
     {{{-1, 0}, {0, 0}, {1, -1}, {1, 0}}}},
    {{{{-1, -1}, {-1, 0}, {0, 0}, {0, 1}}}, // Z
     {{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}}},
    {{{{0, 1}, {0, 0}, {-1, 1}, {-1, 0}}}}, // O
    {{{{-1, 0}, {-1, 1}, {0, -1}, {0, 0}}}, // S
     {{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}}},
    {{{{-1, 1}, {0, -1}, {0, 0}, {0, 1}}}, // L
     {{{-1, 0}, {0, 0}, {1, 0}, {1, 1}}},
     {{{0, -1}, {0, 0}, {0, 1}, {1, -1}}},
     {{{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {0, 2}}}, // I
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};
/*const std::vector<Poly> kBlocks[] = {
    {{{{1, 0}, {0, 0}, {0, 1}, {0, -1}}}, // T
     {{{1, 0}, {0, 0}, {-1, 0}, {0, -1}}},
     {{{0, -1}, {0, 0}, {0, 1}, {-1, 0}}},
     {{{1, 0}, {0, 0}, {0, 1}, {-1, 0}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, 1}}}, // J
     {{{-1, 0}, {0, 0}, {1, -1}, {1, 0}}},
     {{{-1, -1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, 0}, {1, 1}}}, // Z
     {{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, -1}, {1, 0}}}}, // O
    {{{{0, 0}, {0, 1}, {1, -1}, {1, 0}}}, // S
     {{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, -1}}}, // L
     {{{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}},
     {{{-1, 1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {0, 0}, {1, 0}, {1, 1}}}},
    {{{{0, -2}, {0, -1}, {0, 0}, {0, 1}}}, // I
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};*/

namespace internal {

using Row = Entry[kM];
using Table = Entry[kN][kM];
using Vis = std::vector<std::array<std::array<int8_t, kM + 2>, kN + 2>>;

void DFS(int g, int x, int y, int t, int mx_t, bool rotate, Vis& vis) {
  vis[g][x][y] = t;
  if (t < mx_t) {
    if (vis[g][x][y-1] > t+1) DFS(g, x, y-1, t+1, mx_t, rotate, vis);
    if (vis[g][x][y+1] > t+1) DFS(g, x, y+1, t+1, mx_t, rotate, vis);
    if (rotate && vis.size() != 1) {
      int g1 = g == (int)vis.size() - 1 ? 0 : g + 1;
      int g2 = g == 0 ? vis.size() - 1 : g - 1;
      if (vis[g1][x][y] > t+1) DFS(g1, x, y, t+1, mx_t, rotate, vis);
      if (vis[g2][x][y] > t+1) DFS(g2, x, y, t+1, mx_t, rotate, vis);
    }
  }
  if (vis[g][x+1][y] > 0) DFS(g, x+1, y, 0, mx_t, rotate, vis);
}

Vis GetVis(Row* input, int kind) {
  Vis ret(kBlocks[kind].size());
  for (size_t g = 0; g < ret.size(); g++) {
    auto& pl = kBlocks[kind][g];
    for (int x = -1; x <= kN; x++) {
      for (int y = -1; y <= kM; y++) {
        bool flag = true;
        for (int i = 0; i < 4; i++) {
          int nx = pl[i].first + x, ny = pl[i].second + y;
          //if (nx < 0 || ny < 0 || nx >= kN || ny >= kM || input[nx][ny])
          if (ny < 0 || nx >= kN || ny >= kM || (nx >= 0 && input[nx][ny])) {
            flag = false;
            break;
          }
        }
        ret[g][x + 1][y + 1] = flag ? 127 : -1;
      }
    }
  }
  return ret;
}

using Weight = std::pair<uint16_t, uint16_t>;

struct Node {
  // dir: 1(down) 2(left) 3(right) 4(rotateL) 5(rotateR)
  int g, x, y, dir;
  Weight w;
  bool operator<(const Node& a) const { return w > a.w; }
};

constexpr int kStartX = 1;
constexpr int kStartY = 4; // 5

Vis Dijkstra(const Vis& v, bool rotate) {
  Vis ret(v.size(), decltype(v[0]){});
  if (v[0][kStartX+1][kStartY+1] < 0) return ret;
  std::vector<std::array<std::array<Weight, kM + 2>, kN + 2>> d(v.size());
  for (auto& i : d) for (auto& j : i) for (auto& k : j) k = {127, 0};
  std::priority_queue<Node> pq;
  pq.push({0, kStartX+1, kStartY+1, 0, {0, 0}});
  d[0][kStartX+1][kStartY+1] = {0, 0};
  while (!pq.empty()) {
    Node nd = pq.top();
    pq.pop();
    if (d[nd.g][nd.x][nd.y] < nd.w) continue;
    ret[nd.g][nd.x][nd.y] = nd.dir;
    Weight wp = {nd.w.first + 1, nd.w.second + 128 + nd.x};
    auto Relax = [&](int g, int x, int y, Weight w, uint8_t dir) {
      if (v[g][x][y] > 0 && w < d[g][x][y]) {
        pq.push({g, x, y, dir, w});
        d[g][x][y] = w;
      }
    };
    Relax(nd.g, nd.x + 1, nd.y, nd.w, 1);
    Relax(nd.g, nd.x, nd.y - 1, wp, 2);
    Relax(nd.g, nd.x, nd.y + 1, wp, 3);
    if (rotate && v.size() != 1) {
      int g1 = nd.g == (int)v.size() - 1 ? 0 : nd.g + 1;
      int g2 = nd.g == 0 ? v.size() - 1 : nd.g - 1;
      Relax(g1, nd.x, nd.y, wp, 4);
      Relax(g2, nd.x, nd.y, wp, 5);
    }
  }
  return ret;
}

void Allowed(void* buf, int kind, bool rotate, int limit, void* ret_buf,
             int len, void* dir_buf, int len_dir) {
  Row* input = (Row*)buf;
  Vis vis = GetVis(input, kind);
  Vis dir;
  if (dir_buf) dir = Dijkstra(vis, rotate);
  if (vis[0][kStartX+1][kStartY+1] != -1) DFS(0, kStartX+1, kStartY+1, 0, limit, rotate, vis);
  Table* ret = (Table*)ret_buf;
  Table* ret_dir = (Table*)dir_buf;
  int G = rotate ? vis.size() : 1;
  for (int g = 0; g < G; g++) {
    for (int x = 1; x <= kN; x++) {
      for (int y = 1; y <= kM; y++) {
        ret[g][x - 1][y - 1] = vis[g][x + 1][y] == -1 && vis[g][x][y] != -1 &&
                               vis[g][x][y] <= limit;
        if (dir_buf) ret_dir[g][x - 1][y - 1] = dir[g][x][y];
      }
    }
  }
  for (int i = kN * kM * G * sizeof(Entry); i < len; i++) {
    ((uint8_t*)ret_buf)[i] = 0;
  }
  if (dir_buf) {
    for (int i = kN * kM * G * sizeof(Entry); i < len_dir; i++) {
      ((uint8_t*)dir_buf)[i] = 0;
    }
  }
}

void Place(void* buf, int kind, int g, int x, int y, int fill) {
  Row* input = (Row*)buf;
  auto& pl = kBlocks[kind][g];
  for (int i = 0; i < 4; i++) {
    int nx = x + pl[i].first, ny = y + pl[i].second;
    if (nx >= kN || ny >= kM || nx < 0 || ny < 0) continue;
    input[x + pl[i].first][y + pl[i].second] = fill;
  }
}

int Remove(void* buf) {
  Row* input = (Row*)buf;
  int i = kN - 1, j = kN - 1;
  for (; i >= 0; i--, j--) {
    bool flag = true;
    for (int y = 0; y < kM; y++) flag &= input[i][y];
    if (flag) {
      j++;
    } else if (i != j) {
      for (int y = 0; y < kM; y++) input[j][y] = input[i][y];
    }
  }
  int ans = j + 1;
  for (; j >= 0; j--) {
    for (int y = 0; y < kM; y++) input[j][y] = 0;
  }
  return ans;
}

std::vector<std::pair<int, int>> Moves(const void* buf, int tot, int g, int x,
                                       int y) {
  std::vector<std::pair<int, int>> ret;
  const Table* dir = (const Table*)buf;
  while (dir[g][x][y]) {
    if (dir[g][x][y] != 1) ret.push_back({x, dir[g][x][y]});
    switch (dir[g][x][y]) {
      case 1: x--; break;
      case 2: y++; break;
      case 3: y--; break;
      case 4: g = g == 0 ? tot - 1 : g - 1; break;
      case 5: g = g == tot - 1 ? 0 : g + 1; break;
      default: x = y = 0;
    }
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

} // namespace internal

#ifdef DEBUG

#include <random>
#include <unistd.h>

std::mt19937_64 gen(2);
using mrand = std::uniform_int_distribution<int>;

int main() {
  for (int i = 0, cnt = 0;; i++) {
    if (i % 100 == 0) printf("%d %d\n", i, cnt);
    int a[kN][kM]{}, b[4][kN][kM]{}, c[4][kN][kM]{};
    for (;; cnt++) {
      int k = mrand(0, 6)(gen);
      internal::Allowed((void*)a, k, false, 8, (void*)b, sizeof(b), (void*)c, sizeof(c));
      int sum = 0;
      for (int i = 0; i < 4; i++) for (int j = 0; j < kN; j++) for (int k = 0; k < kM; k++) sum += b[i][j][k];
      if (!sum) break;
      int d = 0, x = 0, y = 0;
      {
        int cnt = 0, targ = mrand(0, sum - 1)(gen);
        for (int i = 0; i < 4; i++) for (int j = 0; j < kN; j++) for (int k = 0; k < kM; k++) {
          if (cnt <= targ && cnt + b[i][j][k] > targ) d = i, x = j, y = k;
          cnt += b[i][j][k];
        }
      }
      internal::Place((void*)a, k, d, x, y, cnt%9+1);
      assert(internal::Moves(c, kBlocks[k].size(), d, x, y).size() < 100);
      for (int j = 0; j < kN; j++) {
        for (int k = 0; k < kM; k++) printf("%d ", a[j][k]);
        printf("||");
        for (int i = 0; i < 4; i++) {
          for (int k = 0; k < kM; k++) printf("%d ", (int)b[i][j][k]);
          putchar('|');
        }
        puts("");
      }
      puts(""); sleep(1);
      internal::Remove((void*)a);
    }
  }
}

#else

static PyObject* Allowed(PyObject *self, PyObject *args) {
  Py_buffer buf, ret_buf, dir_buf{};
  int rotate, kind, limit;
  // input(int, 20*10), kind(int), rotate(bool), limit(int), ret(int, 4*20*10 or 20*10(rotate))
  if (!PyArg_ParseTuple(args, "w*ipiw*|w*", &buf, &kind, &rotate, &limit, &ret_buf, &dir_buf)) {
    return nullptr;
  }
  long ret_len = kM * kN * sizeof(Entry) * (rotate ? kBlocks[kind].size() : 1);
  if (buf.len < (long)(kM * kN * sizeof(Entry)) ||
      ret_buf.len < ret_len || (dir_buf.buf && dir_buf.len < ret_len)) {
    PyErr_SetString(PyExc_ValueError, "incorrect buffer length");
    PyBuffer_Release(&buf);
    PyBuffer_Release(&ret_buf);
    if (dir_buf.buf) PyBuffer_Release(&dir_buf);
    return nullptr;
  }
  internal::Allowed(buf.buf, kind, rotate, limit, ret_buf.buf, ret_buf.len,
                    dir_buf.buf, dir_buf.len);
  PyBuffer_Release(&buf);
  PyBuffer_Release(&ret_buf);
  if (dir_buf.buf) PyBuffer_Release(&dir_buf);
  Py_RETURN_NONE;
}

static PyObject* Place(PyObject *self, PyObject *args) {
  int kind, g, x, y, fill = 1, remove = 0;
  Py_buffer buf;
  // input(int, 20*10), kind(int), g(int), x(int), y(int),
  // fill(int: default=1), remove(bool: default=false)
  if (!PyArg_ParseTuple(args, "w*iiii|ip", &buf, &kind, &g, &x, &y, &fill, &remove)) {
    return nullptr;
  }
  internal::Place(buf.buf, kind, g, x, y, fill);
  if (remove) {
    int x = internal::Remove(buf.buf);
    PyBuffer_Release(&buf);
    return PyLong_FromLong(x);
  } else {
    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
  }
}

static PyObject* Remove(PyObject *self, PyObject *args) {
  Py_buffer buf;
  // input(int, 20*10)
  if (!PyArg_ParseTuple(args, "w*", &buf)) return nullptr;
  int x = internal::Remove(buf.buf);
  PyBuffer_Release(&buf);
  return PyLong_FromLong(x);
}

static PyObject* Moves(PyObject *self, PyObject *args) {
  Py_buffer buf;
  int kind, rotate, g, x, y;
  // input(int, 20*10), kind(int), rotate(bool), g(int), x(int), y(int)
  if (!PyArg_ParseTuple(args, "w*iiiii", &buf, &kind, &rotate, &g, &x, &y)) {
    return nullptr;
  }
  std::vector<std::pair<int, int>> trace =
      internal::Moves(buf.buf, rotate ? kBlocks[kind].size() : 1, g, x, y);
  PyBuffer_Release(&buf);
  PyObject* ret = PyList_New(trace.size());
  for (size_t i = 0; i < trace.size(); i++) {
    PyObject* item = PyTuple_Pack(2, PyLong_FromLong(trace[i].first),
                                  PyLong_FromLong(trace[i].second));
    PyList_SetItem(ret, i, item);
  }
  return ret;
}

static PyMethodDef methods[] = {
  {"Allowed", Allowed, METH_VARARGS, ""},
  {"Place", Place, METH_VARARGS, ""},
  {"Remove", Remove, METH_VARARGS, ""},
  {"Moves", Moves, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};
static PyModuleDef mod = {
  PyModuleDef_HEAD_INIT,
  "Tetris_Internal",
  "",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_Tetris_Internal() {
  return PyModule_Create(&mod);
}

#endif
