import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Aniket's Blog",
  description: "Notes on Machine Learning and Software Engineering",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Posts', link: '/posts/' },
    ],

    sidebar: [
      {
        text: 'Posts',
        items: [
          { 
            text: '2019',
            items: [
              { text: 'Face recognition', link: '/posts/2019/face-recognition' },
              { text: 'Image classification with TF2', link: '/posts/2019/image-classification-with-tf2' },
            ]
          },
          { 
            text: '2020',
            items: [
              { text: 'TF.Data: Build data Pipelines', link: '/posts/2020/tf.data-Creating-Data-Input-Pipelines' },
              { text: 'DCGAN: Generate Fake Celebrity image', link: '/posts/2020/DCGAN' },
            ]
          }
        ]
      }
    ],

    socialLinks: [
      { icon: 'twitter', link: 'https://x.com/aniketmaurya' },
      { icon: 'linkedin', link: 'https://linkedin.com/in/aniketmaurya' },
      { icon: 'github', link: 'https://github.com/aniketmaurya' },

    ]
  }
})
