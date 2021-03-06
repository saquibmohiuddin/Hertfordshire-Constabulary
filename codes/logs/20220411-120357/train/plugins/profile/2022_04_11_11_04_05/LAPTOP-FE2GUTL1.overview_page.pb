?	гY?????гY?????!гY?????	}Xip??$@}Xip??$@!}Xip??$@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:гY?????-!?lV??A8??d?`??YZd;?O??rEagerKernelExecute 0*	?????lb@2K
Iterator::Model::Map??ZӼ???!Z??@?K@)?U???د?1jШZ[E@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!t??uM5@)?]K?=??1???^?2@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2?j+??ݓ?!?3˗?R*@)?j+??ݓ?1?3˗?R*@:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip?q??????!<?|?*E@)?]K?=??1???^?"@:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?<,Ԛ???!???^?#@)?5?;Nс?1?`*F??@:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??0?*x?!nD?a?@)??0?*x?1nD?a?@:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!	??3
@)a2U0*?s?1	??3
@:Preprocessing2F
Iterator::Model??(\?µ?!??L?F?L@)_?Q?k?1?h+t@:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMapr??????!??h+(@)a??+ei?1?P?0? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9}Xip??$@I???qOoV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-!?lV??-!?lV??!-!?lV??      ??!       "      ??!       *      ??!       2	8??d?`??8??d?`??!8??d?`??:      ??!       B      ??!       J	Zd;?O??Zd;?O??!Zd;?O??R      ??!       Z	Zd;?O??Zd;?O??!Zd;?O??b      ??!       JCPU_ONLYY}Xip??$@b q???qOoV@Y      Y@q?߆b?sX@"?

both?Your program is MODERATELY input-bound because 10.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"t10.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQb?97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 