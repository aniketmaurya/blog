---
keywords: Linear Regression, maths, gradient descent, machine learning
description: A tutorial on Linear Regression from scratch in Python
title: Linear Regression from Scratch
toc: true 
badges: true
comments: true
categories: [Machine Learning]
image: https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/440px-Anscombe%27s_quartet_3.svg.png
nb_path: _notebooks/2020-03-27-Linear Regression Scratch.ipynb
layout: notebook
---

<!--
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: _notebooks/2020-03-27-Linear Regression Scratch.ipynb
-->

<div class="container" id="notebook-container">
        
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Import-libraries">Import libraries<a class="anchor-link" href="#Import-libraries"> </a></h2>
</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">random</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
</pre></div>

    </div>
</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Data">Data<a class="anchor-link" href="#Data"> </a></h2>
</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="p">[</span><span class="o">*</span><span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">51</span><span class="p">)]</span>
<span class="n">Y</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">map</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="mi">2</span><span class="o">*</span><span class="n">x</span> <span class="o">+</span> <span class="mi">5</span><span class="p">,</span> <span class="n">X</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

</div>
    {% endraw %}

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Univariate-Regression">Univariate Regression<a class="anchor-link" href="#Univariate-Regression"> </a></h2><p>$h(\theta) = \theta*X + b$</p>
<h3 id="MSE-cost-function">MSE cost function<a class="anchor-link" href="#MSE-cost-function"> </a></h3><p>$\sum (h(x) - y)^2$</p>
<h3 id="Gradient-Descent">Gradient Descent<a class="anchor-link" href="#Gradient-Descent"> </a></h3>
<pre><code>
repeat {

    ?? = ?? - ???J(??) = ?? - LR*1/m * sum((h(??, b) - Y)*X)    

    b = b - ???J(b) =  b - LR*1/m * sum((h(??, b) - Y))
}</code></pre>

</div>
</div>
</div>
    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">mse</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">):</span>
    <span class="n">cost</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">m</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">y_pred</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">m</span><span class="p">):</span>
        <span class="n">cost</span> <span class="o">+=</span> <span class="p">(</span><span class="n">y_pred</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">y_true</span><span class="p">[</span><span class="n">i</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span>
    <span class="k">return</span> <span class="n">cost</span><span class="o">/</span><span class="p">(</span><span class="mi">2</span><span class="o">*</span><span class="n">m</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">der_mse</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">):</span>
    <span class="n">der_cost</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">m</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">y_pred</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">m</span><span class="p">):</span>
        <span class="n">der_cost</span> <span class="o">+=</span> <span class="p">(</span><span class="n">y_pred</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">y_true</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">der_cost</span>

<span class="k">def</span> <span class="nf">predict</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">w</span><span class="o">*</span><span class="n">x</span> <span class="o">+</span> <span class="n">b</span>
</pre></div>

    </div>
</div>
</div>

</div>
    {% endraw %}

    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Intialization of variables</span>

<span class="n">m</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="n">LR</span> <span class="o">=</span> <span class="mf">0.01</span>
<span class="n">w</span><span class="p">,</span><span class="n">b</span> <span class="o">=</span><span class="mi">0</span><span class="p">,</span><span class="mf">0.1</span>

<span class="n">epochs</span> <span class="o">=</span> <span class="mi">10000</span>
<span class="c1"># Training</span>

<span class="n">total_cost</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
    <span class="n">y_pred</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">epoch_cost</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">num</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">)):</span>
        <span class="n">x</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">data</span>
        <span class="n">y_pred</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="n">y_pred</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">w</span><span class="o">*</span><span class="n">x</span> <span class="o">+</span> <span class="n">b</span><span class="p">)</span>
        
        
        <span class="n">cost</span> <span class="o">=</span> <span class="n">mse</span><span class="p">(</span><span class="n">Y</span><span class="p">[</span><span class="n">num</span><span class="p">:</span><span class="n">num</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">y_pred</span><span class="p">)</span>
        <span class="n">epoch_cost</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">cost</span><span class="p">)</span>
        <span class="n">der_cost</span> <span class="o">=</span> <span class="n">der_mse</span><span class="p">(</span><span class="n">Y</span><span class="p">[</span><span class="n">num</span><span class="p">:</span><span class="n">num</span><span class="o">+</span><span class="mi">1</span><span class="p">],</span> <span class="n">y_pred</span><span class="p">)</span>

        <span class="n">w</span> <span class="o">-=</span> <span class="n">LR</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="n">m</span><span class="p">)</span> <span class="o">*</span> <span class="n">der_cost</span> <span class="o">*</span> <span class="n">x</span>
        <span class="n">b</span> <span class="o">-=</span> <span class="n">LR</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="n">m</span><span class="p">)</span> <span class="o">*</span> <span class="n">der_cost</span>
        
    <span class="n">total_cost</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">epoch_cost</span><span class="p">))</span>
    <span class="k">if</span> <span class="n">i</span><span class="o">%</span><span class="k">500</span>==0:
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;epoch:</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="se">\t\t</span><span class="s1">cost:</span><span class="si">{</span><span class="n">cost</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>epoch:0		cost:0.024546020195931887
epoch:500		cost:0.0035238913511105277
epoch:1000		cost:0.0004771777468473895
epoch:1500		cost:6.461567040474519e-05
epoch:2000		cost:8.749747634800157e-06
epoch:2500		cost:1.1848222450189964e-06
epoch:3000		cost:1.604393419109384e-07
epoch:3500		cost:2.1725438173628743e-08
epoch:4000		cost:2.9418885555175706e-09
epoch:4500		cost:3.983674896607656e-10
epoch:5000		cost:5.3943803161575866e-11
epoch:5500		cost:7.30464704919418e-12
epoch:6000		cost:9.891380608202818e-13
epoch:6500		cost:1.3394131683086816e-13
epoch:7000		cost:1.8137281109430194e-14
epoch:7500		cost:2.4560089530711338e-15
epoch:8000		cost:3.3257381016463754e-16
epoch:8500		cost:4.5034718706313674e-17
epoch:9000		cost:6.09814092196085e-18
epoch:9500		cost:8.25761584212193e-19
</pre>
</div>
</div>

</div>
</div>

</div>
    {% endraw %}

    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">predict</span><span class="p">(</span><span class="mi">2</span><span class="p">),</span> <span class="n">predict</span><span class="p">(</span><span class="mi">9</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>(8.999999990490096, 22.999999991911498)</pre>
</div>

</div>

</div>
</div>

</div>
    {% endraw %}

    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">w</span><span class="p">,</span> <span class="n">b</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>(2.000000000203057, 4.999999990083981)</pre>
</div>

</div>

</div>
</div>

</div>
    {% endraw %}

    {% raw %}
    
<div class="cell border-box-sizing code_cell rendered">
<div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">total_cost</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
    {% endraw %}

</div>
 

