# Changelog

## [0.9.1](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.9.0...v0.9.1) (2025-12-18)


### Documentation

* Fix jupyter notebook steps ([#196](https://github.com/googleapis/langchain-google-spanner-python/issues/196)) ([fffee5d](https://github.com/googleapis/langchain-google-spanner-python/commit/fffee5da8831f127b8870cf667f19007d99d52c3))

## [0.9.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.8.2...v0.9.0) (2025-07-23)


### Features

* Graph_name validation on graph store construction ([#162](https://github.com/googleapis/langchain-google-spanner-python/issues/162)) ([c753a74](https://github.com/googleapis/langchain-google-spanner-python/commit/c753a747f600582f14503d6ed29281a50f997b34))
* **graph:** Enable setting up the timeout for queries ([#178](https://github.com/googleapis/langchain-google-spanner-python/issues/178)) ([1c5cf19](https://github.com/googleapis/langchain-google-spanner-python/commit/1c5cf19f615fe85afbfb94fc30b1ff1112540f5b)), closes [#170](https://github.com/googleapis/langchain-google-spanner-python/issues/170)
* **graph:** Handle cases when a single table is multiplexed by multi… ([#169](https://github.com/googleapis/langchain-google-spanner-python/issues/169)) ([f99c115](https://github.com/googleapis/langchain-google-spanner-python/commit/f99c1153901fddb6f5f28dba89c9a871910bd7db))
* **graph:** Raise InvalidGQLGenerationError with intermediate_steps ([#175](https://github.com/googleapis/langchain-google-spanner-python/issues/175)) ([22f2158](https://github.com/googleapis/langchain-google-spanner-python/commit/22f215813785699c048b48dae33d69aa9b963e15)), closes [#178](https://github.com/googleapis/langchain-google-spanner-python/issues/178)


### Bug Fixes

* **graph:** 'verified_gql' exception ([9940b86](https://github.com/googleapis/langchain-google-spanner-python/commit/9940b86bb2c8a21961f6abf77612afc4c8168f77))
* **graph:** Ensure intermediate_steps always show generated query ([#174](https://github.com/googleapis/langchain-google-spanner-python/issues/174)) ([d74f040](https://github.com/googleapis/langchain-google-spanner-python/commit/d74f0400075f56e9b43880d548a463a5e4ad44a3)), closes [#173](https://github.com/googleapis/langchain-google-spanner-python/issues/173)
* **graph:** Optimize the size of the serialized schema ([#182](https://github.com/googleapis/langchain-google-spanner-python/issues/182)) ([51dba31](https://github.com/googleapis/langchain-google-spanner-python/commit/51dba31fa6b83d5a36ffa1d7f605fe1f01dd8786)), closes [#181](https://github.com/googleapis/langchain-google-spanner-python/issues/181)
* **graph:** Sorts keys to make serialization determistic ([#184](https://github.com/googleapis/langchain-google-spanner-python/issues/184)) ([cc156f3](https://github.com/googleapis/langchain-google-spanner-python/commit/cc156f3c9ecc569f3792a12374d0fecfe9fed207)), closes [#183](https://github.com/googleapis/langchain-google-spanner-python/issues/183)


### Documentation

* Correct spelling of Spanner in README.rst ([#179](https://github.com/googleapis/langchain-google-spanner-python/issues/179)) ([a32aade](https://github.com/googleapis/langchain-google-spanner-python/commit/a32aade70a2b6029e7323ecf0498a4b0624e5f13))

## [0.8.2](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.8.1...v0.8.2) (2025-03-12)


### Bug Fixes

* **graph:** Minor tweek to the prompts ([#158](https://github.com/googleapis/langchain-google-spanner-python/issues/158)) ([35a6342](https://github.com/googleapis/langchain-google-spanner-python/commit/35a6342b333e0d307d58b2751b07ba775eebf4a5))
* Handle unspecified column_configs for KNN ([#157](https://github.com/googleapis/langchain-google-spanner-python/issues/157)) ([2109892](https://github.com/googleapis/langchain-google-spanner-python/commit/2109892dfc0deac923fc413bf8527c588c61a3d6))


### Documentation

* **samples:** Provide an end to end notebook for graphrag using custom retrievers ([#159](https://github.com/googleapis/langchain-google-spanner-python/issues/159)) ([7ff3620](https://github.com/googleapis/langchain-google-spanner-python/commit/7ff362068288841e5c6f5cdb58038539022e75fe))

## [0.8.1](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.8.0...v0.8.1) (2025-02-20)


### Bug Fixes

* **graph:** Allow property names with different cases ([#149](https://github.com/googleapis/langchain-google-spanner-python/issues/149)) ([37f2324](https://github.com/googleapis/langchain-google-spanner-python/commit/37f2324c37c83bcbbbeb7e04e34337c1d11edbe9))
* **graph:** Support DATE type ([#150](https://github.com/googleapis/langchain-google-spanner-python/issues/150)) ([941e38e](https://github.com/googleapis/langchain-google-spanner-python/commit/941e38eece81612dfd42f7e5343e6f3691605fa9))

## [0.8.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.7.0...v0.8.0) (2025-02-07)


### Features

* **graph:** Flexible Schema Extension to SpannerGraphStore ([#125](https://github.com/googleapis/langchain-google-spanner-python/issues/125)) ([44db678](https://github.com/googleapis/langchain-google-spanner-python/commit/44db67837dd81344fecfe068c2036d5a5345aecf))
* Implement Approximate Nearest Neighbor support for DDL (CREATE TABLE, CREATE VECTOR INDEX) ([#124](https://github.com/googleapis/langchain-google-spanner-python/issues/124)) ([5a25f91](https://github.com/googleapis/langchain-google-spanner-python/commit/5a25f91d5e96e19fc7b05b50fc98b79baa8b8f9e))
* **samples:** Provide guide for ANN vector store end-to-end usage in Jupyter Notebook ([#139](https://github.com/googleapis/langchain-google-spanner-python/issues/139)) ([f78b9ee](https://github.com/googleapis/langchain-google-spanner-python/commit/f78b9ee37497876946745f5ed78c2c62b185b3eb)), closes [#94](https://github.com/googleapis/langchain-google-spanner-python/issues/94)


### Bug Fixes

* Make ANN algorithm updates based off usage + testing ([#140](https://github.com/googleapis/langchain-google-spanner-python/issues/140)) ([524678b](https://github.com/googleapis/langchain-google-spanner-python/commit/524678b3038e61d73fb49469a678ef16cdf8ae7c))
* **testing+linting:** Add nox lint+format directives ([#123](https://github.com/googleapis/langchain-google-spanner-python/issues/123)) ([b10dc28](https://github.com/googleapis/langchain-google-spanner-python/commit/b10dc28ac0f30b0907be2c0747c1890d4d0ba034))

## [0.7.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.6.0...v0.7.0) (2025-01-29)


### Features

* **graph:** Add Custom Retrievers for Spanner Graph RAG. ([#122](https://github.com/googleapis/langchain-google-spanner-python/issues/122)) ([bf2903a](https://github.com/googleapis/langchain-google-spanner-python/commit/bf2903a2a12910d97503a6032bf413ddafe256cf))


### ⚠ BREAKING CHANGES

* extract_gql, fix_gql_syntax are now in the graph_utils module([#122](https://github.com/googleapis/langchain-google-spanner-python/issues/122)) ([bf2903a](https://github.com/googleapis/langchain-google-spanner-python/commit/bf2903a2a12910d97503a6032bf413ddafe256cf))

## [0.6.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.5.0...v0.6.0) (2024-12-05)


### Features

* Add Spanner Graph QA Chain ([#111](https://github.com/googleapis/langchain-google-spanner-python/issues/111)) ([e22abde](https://github.com/googleapis/langchain-google-spanner-python/commit/e22abde9a94625ee69f8975fc0950cedd11bc542))

## [0.5.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.4.1...v0.5.0) (2024-11-25)


### Features

* **graph:** Add Spanner Graph support to LangChain GraphStore interface ([#104](https://github.com/googleapis/langchain-google-spanner-python/issues/104)) ([98c2f8f](https://github.com/googleapis/langchain-google-spanner-python/commit/98c2f8f395e71813f6b1d59e2dedb8c053fee7eb))

## [0.4.1](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.4.0...v0.4.1) (2024-10-04)


### Bug Fixes

* Adding support to initialize with empty metadata columns ([#99](https://github.com/googleapis/langchain-google-spanner-python/issues/99)) ([3a8367c](https://github.com/googleapis/langchain-google-spanner-python/commit/3a8367c82705f352cb263ebfed30da02977de4cc))

## [0.4.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.3.0...v0.4.0) (2024-09-24)


### Features

* Remove support for Python 3.8 ([#92](https://github.com/googleapis/langchain-google-spanner-python/issues/92)) ([8446e0e](https://github.com/googleapis/langchain-google-spanner-python/commit/8446e0e68b5a86d9ad96925908159aa4c5e9b484))


### Bug Fixes

* Use Spanner's UPSERT over INSERT ([#90](https://github.com/googleapis/langchain-google-spanner-python/issues/90)) ([2637e2d](https://github.com/googleapis/langchain-google-spanner-python/commit/2637e2de2ab75dfd51abff7cf0b0c5cd90e6cec9))


### Documentation

* Fix format in README.rst ([#84](https://github.com/googleapis/langchain-google-spanner-python/issues/84)) ([366a040](https://github.com/googleapis/langchain-google-spanner-python/commit/366a040828fdecc28217661955d9c6376808cc9c))

## [0.3.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.2.1...v0.3.0) (2024-05-06)


### Features

* **ci:** Test against multiple versions ([#45](https://github.com/googleapis/langchain-google-spanner-python/issues/45)) ([1e9c1a1](https://github.com/googleapis/langchain-google-spanner-python/commit/1e9c1a1fcadc85f5f45837cdef1261c697cc89f7))


### Bug Fixes

* Make client optional when init ([#60](https://github.com/googleapis/langchain-google-spanner-python/issues/60)) ([0f5124a](https://github.com/googleapis/langchain-google-spanner-python/commit/0f5124a97b6d7c6fcba13bf22b4e01b41d62d347))


### Documentation

* Add API reference docs ([#59](https://github.com/googleapis/langchain-google-spanner-python/issues/59)) ([0f62a6a](https://github.com/googleapis/langchain-google-spanner-python/commit/0f62a6af8399349da06a366d8a29f792a7bcf049))
* Add github links ([#37](https://github.com/googleapis/langchain-google-spanner-python/issues/37)) ([da7f833](https://github.com/googleapis/langchain-google-spanner-python/commit/da7f833aec89f176379f18f16ef5bc069b5470e0))

## [0.2.1](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.2.0...v0.2.1) (2024-03-06)


### Bug Fixes

* Update user agent ([#36](https://github.com/googleapis/langchain-google-spanner-python/issues/36)) ([a8f6f71](https://github.com/googleapis/langchain-google-spanner-python/commit/a8f6f71e9e2567f03d1428cf7c76304f4fa4aa8e))


### Documentation

* Update colabs ([#30](https://github.com/googleapis/langchain-google-spanner-python/issues/30)) ([af66f7c](https://github.com/googleapis/langchain-google-spanner-python/commit/af66f7c13b0e9a972718e57ce64ce73348035940))

## [0.2.0](https://github.com/googleapis/langchain-google-spanner-python/compare/v0.1.0...v0.2.0) (2024-02-29)


### Features

* Creating static utility for initialing chat history table ([#26](https://github.com/googleapis/langchain-google-spanner-python/issues/26)) ([e61499b](https://github.com/googleapis/langchain-google-spanner-python/commit/e61499b8146b2050e6ce7a59e8fc2d3496e77eff))


### Documentation

* Adding codelab for Spanner with DocLoader, VectorStore  & Memory ([#23](https://github.com/googleapis/langchain-google-spanner-python/issues/23)) ([06a6c95](https://github.com/googleapis/langchain-google-spanner-python/commit/06a6c95a01184e712ffda3a74fbe1cc22c495297))
* Update Sample Netflix Application ([#28](https://github.com/googleapis/langchain-google-spanner-python/issues/28)) ([2508ca8](https://github.com/googleapis/langchain-google-spanner-python/commit/2508ca8cb28e277fa538db842d9d35ed60b4db44))

## 0.1.0 (2024-02-26)


### Features

* Add document loader ([#9](https://github.com/googleapis/langchain-google-spanner-python/issues/9)) ([7a77d26](https://github.com/googleapis/langchain-google-spanner-python/commit/7a77d2616e2feacd7130852adb6e5d2aaab81da2))
* Spanner Implementation for Vector Store ([#10](https://github.com/googleapis/langchain-google-spanner-python/issues/10)) ([9cbae82](https://github.com/googleapis/langchain-google-spanner-python/commit/9cbae82b4c2093344071124b08f1a745a77580a7))
* SpannerChatMessageHistory implementation ([#7](https://github.com/googleapis/langchain-google-spanner-python/issues/7)) ([f9a3b93](https://github.com/googleapis/langchain-google-spanner-python/commit/f9a3b931dd61079ddb16f410ab2f9c47bde623ea))


### Documentation

* Update README.md ([#22](https://github.com/googleapis/langchain-google-spanner-python/issues/22)) ([e9eb86b](https://github.com/googleapis/langchain-google-spanner-python/commit/e9eb86babba490fd0dbb19e67ad50603d5959615))
* Vector store notebook docs ([#18](https://github.com/googleapis/langchain-google-spanner-python/issues/18)) ([4c1bd91](https://github.com/googleapis/langchain-google-spanner-python/commit/4c1bd917db03408058dd5169a8047990590cf43b))
