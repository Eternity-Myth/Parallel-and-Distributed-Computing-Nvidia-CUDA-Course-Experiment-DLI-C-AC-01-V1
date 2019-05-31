# Managing Accelerated Application Memory with CUDA C/C++ Unified Memory and nvprof

## Introduction

## Objectives

## Prerequisites

## SECTION

## Summary

## Advanced Content

TODO:

- move exercises into Jupyter, and then QL
- lesson plan

-----------

- intro apod (or something like it)
- intro nvprof, have them run it on 01-vector-add, which is single threaded and slow
  - highlight pertinent parts of the nvprof output
- exercise: increase the performance of the single threaded kernel (tell it should only take a couple minutes)
  - Some time can be spent here modifying the execution configuration and looking at the performance changes
- walk through my experiements, multi thread, multi blocks, smaller blocks,  and then...
- Intro SMs
- exercise: get device info, and also SM blocks (make them use the docs for this)
- exercise: make SM blocks
- Intro UM - then progress through the process as it stands, highlighting details as the need arises
  - exercise: mallocManaged, then, access on CPU, access on GPU, CPU then GPU, GPU then CPU, more?
- Init in kernel to see actual kernel run time, highlight page fault changes

- Intro mem prefetching
  - exercise: prefetch one of the arrays, then 2, then 3, checking nvprof output at each
  - exercise?: prefetch to cpu (I think yes)

- exercise: Saxpy
  - Ask them to work iteratively
  - Make it single threaded to start
  - Add a bug or two maybe
