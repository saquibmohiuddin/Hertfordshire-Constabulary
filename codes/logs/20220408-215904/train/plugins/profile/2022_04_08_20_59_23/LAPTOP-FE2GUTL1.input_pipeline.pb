	J{?/L???J{?/L???!J{?/L???	?[}B?2@?[}B?2@!?[}B?2@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:J{?/L???S??:??A?C??????Y?ZӼ???rEagerKernelExecute 0*	?????,f@2K
Iterator::Model::Mapq???h ??!?+?E?Q@)??6???1k?k?N@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?q??????!??^?1@)9??v????1??%?O-@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?
F%u??!Ne?$Ǣ@)?
F%u??1Ne?$Ǣ@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?]K?=??!??b&?=@)?q?????1??^?@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg??j+???!???!?a@)y?&1?|?1??g(,?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Lu?!d
(s@)??_?Lu?1d
(s@:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU???N@s?!?0!?1@)U???N@s?1?0!?1@:Preprocessing2F
Iterator::Modelz?):?˿?!?Rg???Q@)a??+ei?1???#????:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?ZӼ???!??}O @){?G?zd?1%??D???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?[}B?2@I
?`?Z|T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S??:??S??:??!S??:??      ??!       "      ??!       *      ??!       2	?C???????C??????!?C??????:      ??!       B      ??!       J	?ZӼ????ZӼ???!?ZӼ???R      ??!       Z	?ZӼ????ZӼ???!?ZӼ???b      ??!       JCPU_ONLYY?[}B?2@b q
?`?Z|T@