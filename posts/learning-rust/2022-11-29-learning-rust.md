---
title: 'My first day of learning Rust 🦀'
description: "Documentation of learning Rust"
badges: true
categories:
- rust
- notes
date: '2022-11-29'
keywords: programming, coding, rust
layout: post
toc: true
---

I have been using Python since my college (2018), it has allowed me to experiment my ideas quickly without worrying a lot about syntax and data structures.
I could quickly implement complex algorithm and do some operations in a single line that would take more than 3 LOC in other languages like Java.

```python
# swapping variable in Python
a = 1
b = 2

a, b = b, a
```

Today, I am learning Rust. I am particularly interested in this programming language because of its use cases in wide areas like optimized image resize and creating optimized programs and binding it with other language like Python.

Now let's talk about Rust 🦀

**How to create a dynamic string?**

We can compare it with Python staticmethod `SomeClass.somemethod`, with `::` instead of `.`.

> The double colon :: operator allows us to namespace this particular from function under the String type rather than using some sort of name like string_from. (source)[https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html#the-string-type]

```rust
let s = String::from("hello");
```

This string is mutable and to add more characters we can do the following -
```rust
    s.push_str(", world!");
```

<!-- ## Memory Management in Rust (Owenership) -->
