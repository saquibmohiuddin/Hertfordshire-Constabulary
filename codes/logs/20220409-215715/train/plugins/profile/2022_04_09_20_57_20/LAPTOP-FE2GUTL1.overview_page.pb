?	???9=?????9=??!???9=??	?.?:?8*@?.?:?8*@!?.?:?8*@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???9=??????[??A??:q9???Yi㈵???rEagerKernelExecute 0*	̡E???c@2K
Iterator::Model::Map#/kb???!%??|??P@)??rf?B??1?=~?mL@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?v?$??!??????/@)@OI???1:???̥+@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2???????!C΂?1?$@)???????1C΂?1?$@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?c???H??!P?o?"@)
?F???1??vHY@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip??t????!??tE!>@)~?.rOw?1F??O}@:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceW?????t?!N?-S	@)W?????t?1N?-S	@:Preprocessing2F
Iterator::Model?wE𿕼?!????wQ@)bMeQ?Eq?1
???$@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:!t?%l?!	DH3@):!t?%l?1	DH3@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapOI?V??!??ڥ0%@)n?2d?a?1??䳱y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?.?:?8*@I :????U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????[??????[??!????[??      ??!       "      ??!       *      ??!       2	??:q9?????:q9???!??:q9???:      ??!       B      ??!       J	i㈵???i㈵???!i㈵???R      ??!       Z	i㈵???i㈵???!i㈵???b      ??!       JCPU_ONLYY?.?:?8*@b q :????U@Y      Y@q???=?W@"?

both?Your program is MODERATELY input-bound because 13.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"t11.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?95.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 