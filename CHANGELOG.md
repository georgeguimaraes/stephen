# Changelog

## 1.0.0 (2026-01-12)


### Features

* add ColBERTv2 features: query padding, batch encoding, compression, PLAID ([20ce59b](https://github.com/georgeguimaraes/stephen/commit/20ce59b81eea986c0c1f2802994a3c827db6daf0))
* add KMeans module and index delete/update operations ([40e5793](https://github.com/georgeguimaraes/stephen/commit/40e57936d7f75e397ee3d2ba9e40194f4de1f853))
* add multi-vector batch query support ([32398b2](https://github.com/georgeguimaraes/stephen/commit/32398b2e10e38be1df7cfcebb2b578a9c71cc8d8))
* add multi-vector query fusion functions ([c61d30d](https://github.com/georgeguimaraes/stephen/commit/c61d30d5d4be54bc1fd8904a6140242b1ef31d80))
* add official ColBERT model loading support ([a4ef306](https://github.com/georgeguimaraes/stephen/commit/a4ef306fc9971773933e36becd4359f1a3a979b3))
* add pseudo-relevance feedback (PRF) for query expansion ([847bcc1](https://github.com/georgeguimaraes/stephen/commit/847bcc19320d134a28151f42d10a41daf003d0da))
* add query-document visualization for debugging ([5acbcd8](https://github.com/georgeguimaraes/stephen/commit/5acbcd8b9488035e8fd529d9956b08555b60f4a6))
* add rerank_texts for index-free reranking ([5594f67](https://github.com/georgeguimaraes/stephen/commit/5594f676ac5d065ee08062223b704f29e13b5891))
* add score normalization functions ([f1da0bd](https://github.com/georgeguimaraes/stephen/commit/f1da0bd312a6f8f3412ad5e2002b29446737ea4c))
* auto-detect ColBERT base model type from config.json ([7e88942](https://github.com/georgeguimaraes/stephen/commit/7e88942377d98db1ee1c7dab423c68d2f884e583))
* complete Phase 3 with Python ColBERT feature parity ([b80abfb](https://github.com/georgeguimaraes/stephen/commit/b80abfbcf36ae90714fca53478651f03137baaf0))
* implement ColBERT-style neural retrieval library ([e055b22](https://github.com/georgeguimaraes/stephen/commit/e055b22b01076c1a16e10aad459e961b1c35c68c))


### Bug Fixes

* correct type specs and configure EXLA for tests ([6f887c8](https://github.com/georgeguimaraes/stephen/commit/6f887c8e65c652880f08e7648f608ced86c1770e))
* remove unsupported Electra from model_type mapping ([2a5e658](https://github.com/georgeguimaraes/stephen/commit/2a5e6588f92addd323688f1c89d88905be81547b))
