	??0?*????0?*??!??0?*??	???z&M!@???z&M!@!???z&M!@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??0?*??e?`TR'??AaTR'????Y?!??u???rEagerKernelExecute 0*	????̬g@2K
Iterator::Model::Map?Zd;߿?!S?)?nP@)??s????1?????L@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeatΈ?????!???3@)T㥛? ??18??ߧ?0@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2ŏ1w-!??!SP?? @)ŏ1w-!??1SP?? @:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???_vO??!???A@)U???N@??1?H?w'?@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip????o??!?JZ?5@@)_?Q?{?1?=tf??@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?+e?Xw?!}x?v@)?+e?Xw?1}x?v@:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOv?![???@)??_vOv?1[???@:Preprocessing2F
Iterator::Model????Mb??!???<?P@)y?&1?l?1d0?p*???:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap?5?;Nё?!?N%?_"@)??_?Le?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s6.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???z&M!@I???0[?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?`TR'??e?`TR'??!e?`TR'??      ??!       "      ??!       *      ??!       2	aTR'????aTR'????!aTR'????:      ??!       B      ??!       J	?!??u????!??u???!?!??u???R      ??!       Z	?!??u????!??u???!?!??u???b      ??!       JCPU_ONLYY???z&M!@b q???0[?V@