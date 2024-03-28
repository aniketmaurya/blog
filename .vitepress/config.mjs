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
              { text: 'Deploy Machine Learning Web Apps for Free', link: '/posts/2020/deploy-python-heroku.md' },
              
            ]
          },
          {
            text: '2021',
            items: [
              { text: 'Pix2Pix - Image to image translation', link: '/posts/2021/Pix2Pix explained with code' },
            ]
          },
          {
            text: '2022',
            items: [
              { text: 'HappyWhale 🐳: PyTorch Training from scratch Lite', link: '/posts/2022/happywhale-pytorch-training-from-scratch-lite' },
            ]
          },
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
