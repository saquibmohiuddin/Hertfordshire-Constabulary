	?u??????u?????!?u?????	????o?'@????o?'@!????o?'@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?u???????%䃞??A?3??7??Y??S㥛??rEagerKernelExecute 0*effff?h@)       =2K
Iterator::Model::Maps??A??!M?_{ȥN@)5?8EGr??1??G
&?H@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat????ׁ??!?=??b 3@)X?5?;N??1??l??0@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?z6?>??!?^ĉ?&@)?z6?>??1?^ĉ?&@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatetF??_??!K/?D?'@)y?&1???1?????@:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen????!)??A??@)n????1)??A??@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip???????!???\VLB@)?J?4??1?ˊ??@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor"??u??q?!^K/?D@)"??u??q?1^K/?D@:Preprocessing2F
Iterator::Model??0?*??!Jx???O@)?J?4q?1?ˊ?? @:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapF%u???!???ˊ?*@)??_?Le?1?]K/???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s8.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????o?'@I??RV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??%䃞????%䃞??!??%䃞??      ??!       "      ??!       *      ??!       2	?3??7???3??7??!?3??7??:      ??!       B      ??!       J	??S㥛????S㥛??!??S㥛??R      ??!       Z	??S㥛????S㥛??!??S㥛??b      ??!       JCPU_ONLYY????o?'@b q??RV@