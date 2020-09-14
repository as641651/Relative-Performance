# Relative-Performance

Assigns relative scores to a set of equivalent algorithms.

### Input

Measurements of execution times of all equivalent algorithms in JSON format

```python
timings.json

{
  "Expression1":{
                  "0": [ 0.12, 0.23, ..., 0.19 ],
                  "2": [ 0.12, 0.23, ..., 0.19 ],
                   ..
                   ..
                  "99": [ 0.12, 0.23, ..., 0.19 ],
                 },
  "Expression2":{
                  "0": [ 0.12, 0.23, ..., 0.19 ],
                  "2": [ 0.12, 0.23, ..., 0.19 ],
                   ..
                   ..
                  "99": [ 0.12, 0.23, ..., 0.19 ],
               },
   ...
 }
```

### Parameters

T:  Number of repetition of the Sort Function
M:  Number of bootstrap iterations in the Compare function
K:  Bootstrap sample size
t: Comparison threshold

### Usage
Example
```bash
python3 relativePerformance.py timings.json -T 30 -M 30 -K 5 -t 0.8
```
