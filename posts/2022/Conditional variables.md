<!-- ---
badges: true
categories:
- python
date: '2022-11-20'
description: Conditional variable and lock in Python
image: https://images.pexels.com/photos/1029635/pexels-photo-1029635.jpeg?auto=compress
keywords: Python
output-file: 2022-11-20 conditional_variables.html
title: 'Conditional variable and lock in Python'
toc: true
draft: true
--- -->
# Conditional variable and lock in Python

```python
from threading import Condition, Lock, Thread
import random
```


```python
found_divisible = False
num = None
cond_var = Condition()
exit_prog = False
```


```python
def divisible(x):
    return x % 5 == 0


def finder():
    global found_divisible, num, exit_prog
    while not exit_prog:
        x = random.randint(1, 1000)
        if divisible(x):
            cond_var.acquire()
            num = x
            found_divisible = True
            cond_var.notify()
            cond_var.release()


def printer():
    global num, found_divisible, exit_prog
    while not exit_prog:
        cond_var.acquire()
        while not found_divisible and not exit_prog:
            cond_var.wait()
        print(num)
        found_divisible = False
        cond_var.release()
```


```python
printerThread = Thread(target=printer)
printerThread.start()

finderThread = Thread(target=finder)
finderThread.start()
```

    525
    735
    930
    580
    100
    880
    320
    780
    720
    265
    965
    740
    375
    40
    835
    505
    120
    290
    985
    720
    745
    175



```python
# Let the threads run for 3 seconds
import time

time.sleep(3)

# Let the threads exit
exit_prog = True

cond_var.acquire()
cond_var.notifyAll()
cond_var.release()

printerThread.join()
finderThread.join()
```

    370
    5
    520
    160
    835
    45
    490
    765
    265
    245
    605
    105
    90
    855
    775
    85
    760
    635
    565
    740
    845
    475
    655
    555
    705
    435
    920
    520
    970
    705
    770
    425
    530
    645
    380
    980
    650
    190
    640
    820
    785
    95
    110
    135
    815
    545
    330
    165
    950
    235
    575
    915
    145
    985
    515
    690
    775
    335
    260
    995
    165
    840
    315
    630
    825
    310
    915
    635
    885
    995
    250
    410
    470
    465
    395
    895
    990
    770
    360
    165
    470
    905
    385
    70
    905
    25
    765
    810
    915
    320
    110
    520
    775
    20
    415
    450
    15
    335
    495
    830
    615
    115
    300
    275
    475
    300
    450
    415
    445
    450
    500
    425
    370
    325
    670
    235
    1000
    650
    135
    340
    365
    605
    280
    520
    415
    675
    965
    180
    150
    580
    985
    85
    790
    835
    215
    550
    275
    285
    145
    5
    595
    340
    420
    885
    15
    695
    555
    890
    280
    630
    955
    550
    505
    470
    95
    595
    915
    440
    740
    335
    960
    400
    980
    245
    645
    995
    385
    190
    285
    585
    230
    330
    520
    915
    320
    225
    250
    755
    670
    995
    140
    430
    125
    195
    135
    745
    475
    655
    850
    295
    675
    735
    945
    155
    850
    335
    940
    235
    30
    370
    75
    55
    765
    645
    420
    940
    495
    520
    235
    410
    480
    465
    625
    610
    310
    615
    810
    355
    285
    290
    490
    350
    635
    785
    450
    235
    785
    80
    440
    90
    615
    365
    255
    310
    320
    90
    700
    85
    860
    950
    475
    670
    60
    15
    705
    30
    150
    810
    105
    920
    10
    40
    820
    610
    305
    75
    170
    135
    280
    150
    640
    405
    120
    410
    490
    575
    655
    580
    910
    545
    605
    25
    205
    195
    305
    145
    325
    850
    135
    505
    315
    835
    320
    555
    110
    250
    510
    745
    645
    915
    690
    835
    310
    385
    430
    450
    20
    30
    615
    330
    930
    685
    255
    130
    540
    305
    995
    440
    715
    585
    65
    870
    210
    640
    55
    720
    895
    410
    875
    600
    615
    400
    890
    210
    285
    740
    140
    675
    560
    360
    665
    500
    925
    545
    495
    590
    610
    320
    935
    710
    795
    405
    395
    80
    60
    610
    880
    955
    665
    840
    790
    715
    100
    875
    740
    675
    695
    645
    400
    25
    245
    560
    320
    940
    670
    165
    450
    610
    480
    220
    20
    560
    45
    550
    750
    135
    615
    65
    890
    120
    405
    90
    575
    495
    200
    590
    140
    310
    645
    685
    735
    10
    640
    400
    540
    155
    630
    240
    785
    300
    895
    920
    875
    645
    815
    345
    360
    925
    95
    890
    580
    630
    945
    255
    610
    260
    930
    885
    340
    300
    955
    830
    235
    890
    105
    40
    460
    315
    305
    955
    990
    395
    30
    850
    275



```python

```
