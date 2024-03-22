---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Aniket's Blog"
  text: "Notes on Machine Learning and Software Engineering"
  tagline: Hi there! I'm Aniket, a Machine Learning - Software Engineer with with over 4 years of experience, demonstrating a strong track record in developing and deploying machine learning models to production.
  actions:
    - theme: brand
      text: Markdown Examples
      link: /markdown-examples
    - theme: alt
      text: API Examples
      link: /api-examples

  image:
    src: https://avatars.githubusercontent.com/u/21018714?s=400&u=b47cb5eb401714a0e38aa8d32ec7031e5c30e346&v=4
    alt: gradient

features:
  - title: Feature A
    details: Lorem ipsum dolor sit amet, consectetur adipiscing elit
  - title: Feature B
    details: Lorem ipsum dolor sit amet, consectetur adipiscing elit
  - title: Feature C
    details: Lorem ipsum dolor sit amet, consectetur adipiscing elit
---


<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #bd34fe 30%, #41d1ff);

  --vp-home-hero-image-background-image: linear-gradient(-45deg, #bd34fe 50%, #47caff 50%);
  --vp-home-hero-image-filter: blur(44px);
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}
</style>
