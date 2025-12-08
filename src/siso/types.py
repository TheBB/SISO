from __future__ import annotations

import numpy as np

type f64 = np.float64
type f32 = np.float32
type i64 = np.int64
type i32 = np.int32
type u32 = np.uint32
type u8 = np.uint8

type f64d = np.dtype[f64]
type f32d = np.dtype[f32]
type i64d = np.dtype[i64]
type i32d = np.dtype[i32]
type u32d = np.dtype[u32]
type u8d = np.dtype[u8]

type Float = f32 | f64 | float
type Int = i64 | i32 | int
type Scalar = Int | Float

type Floatd = f32d | f64d
type Intd = i32d | i64d
type DType = Floatd | Intd


type Array[D: np.dtype] = np.ndarray[tuple[int, ...], D]
type FloatArray = Array[f64d] | Array[f32d]
type IntArray = Array[i64d] | Array[i32d]

type Vector[D: np.dtype] = np.ndarray[tuple[int], D]
type FloatVector = Vector[f64d] | Vector[f32d]
type IntVector = Vector[i64d] | Vector[i32d]

type Matrix[D: np.dtype] = np.ndarray[tuple[int, int], D]
type FloatMatrix = Matrix[f64d] | Matrix[f32d]
type IntMatrix = Matrix[i64d] | Matrix[i32d]
