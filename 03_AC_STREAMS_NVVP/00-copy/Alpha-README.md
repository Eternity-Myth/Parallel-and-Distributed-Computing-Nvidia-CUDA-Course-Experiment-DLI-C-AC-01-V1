- Using the nvvp timeline to identify opportunities for optimization
- Demo pre and post prefetch

- cpu page faults are taking forever, let's init on the gpu
- ask them to do this and compare

- Now we have to wait for all that code to get back to the host, let's prefetch it
- ask them to do this and compare

- Now the timeline shows long memory lag getting data back to host, time to prefetch back to host
- ask them to do this and compare


Streaming Interlude

- Kernel execution and memory transfers occur within CUDA streams
- So far everything has been using the "default stream"
- Should we choose, we can create new streams and perform operations through them
- Operations within a given stream occur in order
- Operations in different streams are not guaranteed to operate in any specific order relative to each other
- The default stream is blocking
- Show basic streaming example before and after in the profiler


- Take a look back at the main timeline thread, notice serial initializations
- ask them to run init in separate streams and compare
