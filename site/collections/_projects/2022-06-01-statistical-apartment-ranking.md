---
date: 2025-04-01 05:20:35 +0300
title: 'Finding the Optimal Apartment: Statistical Ranking Model for Apartment Search'
subtitle: Gaussian Modeling + Scoring 
image: '/images/apartment-ranking.webp'
---
Have you ever been apartment searching, and had a large, comprehensive list of places you were looking at? It can get overwhelming — and is often hard to decide what your best option is because you feel ambivalent towards most if not all apartments on your list (this is especially true in the Bay Area).

I ran into this exact issue in my search, which is why I created a somewhat unconventional method to help point you in the right direction when deciding on a place.

#### This approach assumes the following:

-   You have a list of places you found via search on platforms like  [Airbnb](https://www.airbnb.com/),  [Apartments.com](https://www.apartments.com/),  [Zillow](https://www.zillow.com/), etc.
-   You have the following data on those places:  
    - Monthly rent (optionally with any additional fees, such as utilities)  
    - Unit floor area (in sq. ft)  
    - Rating on your overall feeling  _about the physical unit (excluding floor area)_ on 1 to 5 scale*

In rating each apartment, consider the following: your feelings on safety, management, neighbors, quality of building, etc. All factors should be taken into account. This rating is subjective, which is a quantitative limitation. However, your decision on a place to live is also subjective. This approach aims to mitigate the amount of decision making to be made.

### Method

The method used to quantitatively measure the optimal apartment(s) involves quite a bit of normalization, weighting, and ranking. I did all my calculations in Google Sheets, but I’ll walk through the algorithm here.

Let’s define vectors  _a_ (floor area),  _b_  (rent),  _c_ (your rating), and  _T_ as follows:

$$
a = (\text{area}_1, \text{area}_2, ... \text{area}_n)
$$
$$
b = (\text{rent}_1, \text{rent}_2, ... \text{rent}_n)
$$
$$
c = (\text{feel}_1, \text{feel}_2, ... \text{feel}_n)
$$
$$
T = (a,b,c)
$$

Each component contained in  _a, b,_ and  _c_  are the parameters corresponding to each apartment/place you have listed.  _T_ is simply a collection of those vectors.

Let us rank components within the vectors in  _T_ defined by  _R:_

$$
R = \text{rank}_t(T) = 
\begin{bmatrix}
\text{rank}_a^{-1}(a_1) & \text{rank}_b(b_1) & \text{rank}_c^{-1}(c_1) \\
\vdots & \vdots & \vdots \\
\text{rank}_a^{-1}(a_n) & \text{rank}_b(b_n) & \text{rank}_c^{-1}(c_n)
\end{bmatrix}
$$

You will notice rank is being used in columns with a -1 exponent. This is not a -1 exponent, but rather corresponds to inverse. In our case of ranking, an inverse rank corresponds to ranking in  _descending order. Using descending order for certain parameters allows us to rank those parameters practically. If the “better” values of b are logically higher, then the “better” values of a and c should be lower._

With arbitrary vector  _u,_ we  define the average function as:

$$
\text{avg}(u) = \frac{\sum_{i=1}^{n} u_i}{n}
$$

n is the number of components in u

and use the average function to calculate the averages of row vectors in  _R,_

$$
\overline{\text{rank}(T_i)} = 
\begin{bmatrix}
\text{avg}((R_{11}, R_{12}, R_{13})) \\
\vdots \\
\text{avg}((R_{n1}, R_{n2}, R_{n3}))
\end{bmatrix}
$$

thereby giving us a single column containing the average of area, rent, and personal rating rankings, for each apartment.

Next, we normalize the values in  _a_,  _b,_ and  _c_ according to the normal distribution, using the function  _f(T)_ with  _t_ being each component of  _T._ That is,  _t_ represents each vector containing values for the respective apartment data.

$$
f(T) = \frac{1}{\sigma_t \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{x - \mu_t}{\sigma_t} \right)^2}
$$

Using the normalized data and discrete weights, we can calculate the average of weighted normalized values for each apartment. You should assign values to these weights depending on how important the respective data is. For example, if floor area of the apartment is not as important, you set weight1 for a = 0.5.  **Additionally, regardless of how much weight you assign on rent (_b_), rent should always have a negative weight.** This is because higher rent does not mean better, unless for some reason, you want to spend more on an apartment. If the latter is the case, the magnitude of your assigned weight should be smaller.

$$
\begin{aligned}
w_{1a} &= \text{weight}_1 \leftrightarrow a \\
w_{1b} &= \text{weight}_1 \leftrightarrow b \\
w_{1c} &= \text{weight}_1 \leftrightarrow c
\end{aligned}
$$

$$
\overline{W(f(T))} =
\begin{bmatrix}
\text{avg}((w_{1a} * f(T)_{11},\ w_{1b} * f(T)_{12},\ w_{1c} * f(T)_{13})) \\
\vdots \\
\text{avg}((w_{1a} * f(T)_{n1},\ w_{1b} * f(T)_{n2},\ w_{1c} * f(T)_{n3}))
\end{bmatrix}
$$

Note: weight 1 of a is the discrete weight value corresponding to a, which applies to weight 1 of b and so forth. f(T) is technically a matrix, so we can reference its elements directly.

It should also be significantly noted that your weights should move the values as close to the same decimal position as possible. For example, if all your values of  _b_ are all in the thousands and your values of  _a_ are in the tens, either set the weight of  _b_ to 0.01 or set the weight of  _a_ to 100. Same would be said for  _c._ To reiterate, each “column” should contain values that fall near or within the same decimal place of each other.  **If you do not do this, your normalized values later on will be incredibly skewed and thus any averages will be skewed as well.**

We define a measure  _score1_  as follows, with the weighting rules from above applying here as well:

$$
\begin{aligned}
w_{2a} &= \text{weight}_2 \leftrightarrow a \\
w_{2b} &= \text{weight}_2 \leftrightarrow b \\
w_{2c} &= \text{weight}_2 \leftrightarrow c
\end{aligned}
$$

$$
\text{score}_1 = A =
\begin{bmatrix}
\text{avg}((w_{2a} * a_1,\ w_{2b} * b_1,\ w_{2c} * c_1)) \\
\vdots \\
\text{avg}((w_{2a} * a_n,\ w_{2b} * b_n,\ w_{2c} * c_n))
\end{bmatrix}
$$

See note regarding the weights under the average of weighted normalized values definition.

Now, using the previous derivations, we can calculate a vector giving us a single score derived from all three original parameters: area, rent, and personal rating. Again, we need to use the same weighting rules mentioned earlier.

$$
\begin{aligned}
w_{3a} &= \text{weight}_3 \leftrightarrow a \\
w_{3b} &= \text{weight}_3 \leftrightarrow b \\
w_{3c} &= \text{weight}_3 \leftrightarrow c
\end{aligned}
$$

$$
\text{score}_2 = \text{avg} \left( \left( w_{3a} * \overline{\text{rank}(T_i)},\ w_{3b} * \overline{W(f(T))},\ \text{score}_1 \right) \right)
$$

Then, using that derived score, we again normalize the components in the vector using function  _f_ from earlier, this time in terms of a single arbitrary vector  _u_.

$$
f(u) = \frac{1}{\sigma_u \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{x - \mu_u}{\sigma_u} \right)^2}
$$

Lastly, we use the normalized values of  _score2_  to derive a vector containing the final ranking of apartments:

$$
\text{Apartment rank} =
\begin{bmatrix}
\text{rank}(f(u)_1) \\
\vdots \\
\text{rank}(f(u)_n)
\end{bmatrix}
$$

### Conclusion

The resulting ranking should give you a good idea of what apartments to prioritize, however that may be. Maybe it’s picking which apartments to tour in-person, or maybe it’s choosing which apartments are important enough to check for available listings daily. It’s your choice how you use those rankings, but keep in mind those rankings are based off of the weights you assigned, and thus how important the parameters area, rent, and personal rating are.

Hope you enjoyed this nifty math article, and could make use of this! Let me know what you think, and feel free to reach out should you have any questions.