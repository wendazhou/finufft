# Optimized spreader design

We describe the principal design considerations of the optimized spreader
in the current document.
We recall that the spreading operation (for a fixed kernel and target size) is given
as input a collection of non-uniform points:
```c++
template<typename T, size_t Dim>
struct nu_point_collection {
   size_t num_points;
   std::array<T*, Dim> coordinates;
   T* weights;
};
```

## Spreading subproblem

In this section, we describe the inner-loop kernel of the spreading problem.
It logically implements the spreading operation with no boundary conditions
and no synchronization (i.e. single-threaded).

### Vectorized evaluation of kernels

The main computationally expensive part of the spreading is given by the
evaluation of the spreading kernel.
We first consider the one-dimensional case.
We are required to evaluate a family of polynomials $k_i = p_i(x), i = 1, \dotsc, w$,
where $w$ is the width of the kernel.
By making use of vector instructions, we are able to evaluate $p_i(x)$ for several
different values of $i$ in lockstep in parallel.

We note that the relationship between $w$ and the native width of the vector
instructions of the underlying target (e.g. AVX2, AVX512, NEON) is crucial,
and hence we are required to write a variety of implementations for different $w$.
Indeed, note that although an implementation for $w$ is valid for $w' \leq w$
(by simply zero padding), this may be sub-optimal if excessive padding is required
(e.g. we may miss out on the opportunity to compute the kernel for two points in the
same vector register).

### Subproblem accumulation

After evaluation of the kernel and multiplication by the corresponding strength,
we accumulate the evaluated value into the subproblem buffer.
We again make use of the underlying vector instructions to load, accumulate and store
values from the buffer.

However, we note that an additional difficulty in this case is that the left corner
of the accumulation location may not have the required alignment (even though the buffer itself may).
Indeed, the offset of the left corner is determined in part by the coordinate of the
non-uniform point, and hence may be arbitrary.
In order to optimize the memory operations (and leverage hardware features such as load-to-store forwarding),
we manipulate the vector registers in order to ensure alignment.

### Subproblem size selection

By selecting the tiling of the spreading problem appropriately, we may select
the size of the output buffer for the subproblem (up to some restrictions concerning the required padding).
Empirical results show that when possible, ensuring that the subproblem buffer remains
in the L1 cache of the CPU provides a great uplift in performance, and going over
that limit immediately incurs a non-linear performance penalty due to L1 cache misses.
We thus attempt to ensure whenever possible that our subproblem size is selected to fit
within the L1 cache of the underlying hardware.

We note that in some cases (especially for 3D problems), this may not be possible without
the size of the subproblem target being too small (and causing issues due to the required padding etc.).
Further empirical work is required to determine optimal configurations for such problems.

## Sorting

In order for the spreading to be performed efficiently, it is crucial to identify
the subproblem to which each point belongs, and ensure that they are located contiguously
in memory. This ensure that the subproblem implementation may read efficiently from
the non-uniform point information.

The current implementation attempts to make use of a state-of-the-art parallel comparison
sort `ips4o`.
However, in order to fully leverage this sort, we must repack the data into an array of structure format
and back out.
The packing and unpacking takes about 40% of the total sorting time, but we nonetheless observe
an approximately 3x speed-up over the current bin sort.

A future implementation may consider leveraging a custom parallel counting sort with optimized
data movement.

## Spreading Orchestration

In order to take advantage of multiple CPU cores, we must schedule subproblems on separate
CPU cores.
However, we must do so while minimizing the synchronization cost, as we note that subproblems
are not completely independent.
We describe two strategies: 1) synchronized accumulation and 2) even-odd splitting.
We note that the current implementation is synchronized accumulation, and incurs an approximately
10% overhead compared to a potentially no overhead even-odd splitting implementation.

### Synchronized accumulation

In order to avoid synchronization in the inner loop, we instead accumulate to an
intermediate buffer in the subproblem, and synchronize the accumulation of this buffer
into the main output.
This accumulation is synchronized by blocks (e.g. each thread locks a mutex associated with a block of memory,
currently the block size is set to 4 kiB).
On supported compilers (i.e. c++20 and later), we make use of the lightweight locking mechanism provided by
`std::atomic_wait` (which may e.g. correspond to a `futex` on linux).
If that is not available, we fall back to a multiplexed array of `std::mutex`.

### Even-odd splitting

We note that when blocks are sufficiently large, only neighbouring blocks can have synchronization issues.
It is thus possible to dispatch blocks in a sequence which ensures that no neighbouring blocks are
scheduled at the same time (e.g. in 1D, this corresponds to an even-odd split).
This would enable the subproblem to directly accumulate into the target buffer, rather than an intermediate buffer,
and elide an uneccessary accumulation step.
This could lead to a ~10% speedup of spreading for problems of density ~1, with diminishing returns for problems
of higher density (as the cost of the subproblem increases in relation to the cost of the accumulation).
