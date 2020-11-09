#define PY_SSIZE_T_CLEAN
#include <cstdint>
#include <array>
#include <vector>
#include <utility>
//#include <python3.8/Python.h>
#include <Python.h>

const int kN = 20, kM = 10;
using Entry = int32_t;
using Row = Entry[kM];
using Table = Entry[kN][kM];
using Poly = std::array<std::pair<int, int>, 4>;
using Vis = std::vector<std::array<std::array<int8_t, kM + 2>, kN + 2>>;

const std::vector<Poly> kBlocks[] = {
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
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};

namespace internal {

void DFS(int g, int x, int y, int t, int mx_t, bool rotate, Vis& vis) {
  vis[g][x][y] = t;
  if (t < mx_t) {
    int g1 = g == (int)vis.size() - 1 ? 0 : g + 1;
    int g2 = g == 0 ? vis.size() - 1 : g - 1;
    if (vis[g][x][y-1] > t+1) DFS(g, x, y-1, t+1, mx_t, rotate, vis);
    if (vis[g][x][y+1] > t+1) DFS(g, x, y+1, t+1, mx_t, rotate, vis);
    if (rotate) {
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
        //printf("!%d %d %d %d\n", (int)g, x, y, (int)flag);
      }
    }
  }
  return ret;
}

void Allowed(void* buf, int kind, bool rotate, int limit, void* ret_buf, int len) {
  Row* input = (Row*)buf;
  Vis vis = GetVis(input, kind);
  if (vis[0][0+1][5+1] != -1) DFS(0, 0+1, 5+1, 0, limit, rotate, vis);
  if (rotate) {
    Table* ret = (Table*)ret_buf;
    for (int g = 0; g < (int)vis.size(); g++) {
      for (int x = 1; x <= kN; x++) {
        for (int y = 1; y <= kM; y++) {
          ret[g][x - 1][y - 1] = vis[g][x + 1][y] == -1 && vis[g][x][y] != -1 &&
                                 vis[g][x][y] <= limit;
        }
      }
    }
    for (int i = kN * kM * vis.size() * sizeof(Entry); i < len; i++) {
      ((uint8_t*)ret_buf)[i] = 0;
    }
  } else {
    Row* ret = (Row*)ret_buf;
    for (int x = 1; x <= kN; x++) {
      for (int y = 1; y <= kM; y++) {
        ret[x - 1][y - 1] = vis[0][x + 1][y] == -1 && vis[0][x][y] != -1 &&
                            vis[0][x][y] <= limit;
      }
    }
    for (int i = kN * kM * sizeof(Entry); i < len; i++) {
      ((uint8_t*)ret_buf)[i] = 0;
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

} // namespace internal

#ifdef DEBUG

#include <random>
#include <unistd.h>

std::mt19937_64 gen;
using mrand = std::uniform_int_distribution<int>;

int main() {
  for (int i = 0, cnt = 0;; i++) {
    if (i % 100 == 0) printf("%d %d\n", i, cnt);
    int a[20][10]{}, b[4][20][10]{};
    for (;; cnt++) {
      int k = mrand(0, 6)(gen);
      internal::Allowed((void*)a, k, true, 2, (void*)b, sizeof(b));
      int sum = 0;
      for (int i = 0; i < 4; i++) for (int j = 0; j < 20; j++) for (int k = 0; k < 10; k++) sum += b[i][j][k];
      if (!sum) break;
      int d = 0, x = 0, y = 0;
      {
        int cnt = 0, targ = mrand(0, sum - 1)(gen);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 20; j++) for (int k = 0; k < 10; k++) {
          if (cnt <= targ && cnt + b[i][j][k] > targ) d = i, x = j, y = k;
          cnt += b[i][j][k];
        }
      }
      internal::Place((void*)a, k, d, x, y, 1);
      /*for (int j = 0; j < 20; j++) {
        for (int k = 0; k < 10; k++) printf("%d ", a[j][k]);
        printf("||");
        for (int i = 0; i < 4; i++) {
          for (int k = 0; k < 10; k++) printf("%d ", (int)b[i][j][k]);
          putchar('|');
        }
        puts("");
      }
      puts(""); sleep(1);*/
      internal::Remove((void*)a);
    }
  }
}

#else

static PyObject* Allowed(PyObject *self, PyObject *args) {
  Py_buffer buf, ret_buf;
  int rotate, kind, limit;
  // input(int, 20*10), kind(int), rotate(bool), limit(int), ret(int, 4*20*10 or 20*10(rotate))
  if (!PyArg_ParseTuple(args, "w*ipiw*", &buf, &kind, &rotate, &limit, &ret_buf)) {
    return nullptr;
  }
  if (buf.len < (long)(kM * kN * sizeof(Entry)) ||
      ret_buf.len < (long)(kM * kN * sizeof(Entry) * (rotate ? kBlocks[kind].size() : 1))) {
    PyErr_SetString(PyExc_ValueError, "incorrect buffer length");
    return nullptr;
  }
  internal::Allowed(buf.buf, kind, rotate, limit, ret_buf.buf, ret_buf.len);
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
    return PyLong_FromLong(internal::Remove(buf.buf));
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject* Remove(PyObject *self, PyObject *args) {
  Py_buffer buf;
  // input(int, 20*10)
  if (!PyArg_ParseTuple(args, "w*", &buf)) return nullptr;
  return PyLong_FromLong(internal::Remove(buf.buf));
}

static PyMethodDef methods[] = {
  {"Allowed", Allowed, METH_VARARGS, ""},
  {"Place", Place, METH_VARARGS, ""},
  {"Remove", Remove, METH_VARARGS, ""},
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
