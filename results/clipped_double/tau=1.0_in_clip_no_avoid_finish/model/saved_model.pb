מ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628�
�
custom_model_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namecustom_model_1/dense_3/bias
�
/custom_model_1/dense_3/bias/Read/ReadVariableOpReadVariableOpcustom_model_1/dense_3/bias*
_output_shapes
:	*
dtype0
�
custom_model_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�	*.
shared_namecustom_model_1/dense_3/kernel
�
1custom_model_1/dense_3/kernel/Read/ReadVariableOpReadVariableOpcustom_model_1/dense_3/kernel*
_output_shapes
:	�	*
dtype0
�
custom_model_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecustom_model_1/dense_2/bias
�
/custom_model_1/dense_2/bias/Read/ReadVariableOpReadVariableOpcustom_model_1/dense_2/bias*
_output_shapes	
:�*
dtype0
�
custom_model_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namecustom_model_1/dense_2/kernel
�
1custom_model_1/dense_2/kernel/Read/ReadVariableOpReadVariableOpcustom_model_1/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
custom_model_1/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecustom_model_1/conv2d_5/bias
�
0custom_model_1/conv2d_5/bias/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_5/bias*
_output_shapes
:@*
dtype0
�
custom_model_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name custom_model_1/conv2d_5/kernel
�
2custom_model_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
�
custom_model_1/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecustom_model_1/conv2d_4/bias
�
0custom_model_1/conv2d_4/bias/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_4/bias*
_output_shapes
:@*
dtype0
�
custom_model_1/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name custom_model_1/conv2d_4/kernel
�
2custom_model_1/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_4/kernel*&
_output_shapes
: @*
dtype0
�
custom_model_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecustom_model_1/conv2d_3/bias
�
0custom_model_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_3/bias*
_output_shapes
: *
dtype0
�
custom_model_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name custom_model_1/conv2d_3/kernel
�
2custom_model_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpcustom_model_1/conv2d_3/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������TT*
dtype0*$
shape:���������TT
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1custom_model_1/conv2d_3/kernelcustom_model_1/conv2d_3/biascustom_model_1/conv2d_4/kernelcustom_model_1/conv2d_4/biascustom_model_1/conv2d_5/kernelcustom_model_1/conv2d_5/biascustom_model_1/dense_2/kernelcustom_model_1/dense_2/biascustom_model_1/dense_3/kernelcustom_model_1/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_113065734

NoOpNoOp
�)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�(
value�(B�( B�(
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
flatten

dense1

dense2

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
 &_jit_compiled_convolution_op*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

kernel
bias
 -_jit_compiled_convolution_op*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

kernel
bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias*

Gserving_default* 
^X
VARIABLE_VALUEcustom_model_1/conv2d_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEcustom_model_1/conv2d_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcustom_model_1/conv2d_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEcustom_model_1/conv2d_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcustom_model_1/conv2d_5/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEcustom_model_1/conv2d_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcustom_model_1/dense_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcustom_model_1/dense_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcustom_model_1/dense_3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcustom_model_1/dense_3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
	1

2
3
4
5*
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
* 

0
1*

0
1*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
* 

0
1*

0
1*
* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
* 
* 
* 
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

btrace_0* 

ctrace_0* 

0
1*

0
1*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 

0
1*

0
1*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecustom_model_1/conv2d_3/kernelcustom_model_1/conv2d_3/biascustom_model_1/conv2d_4/kernelcustom_model_1/conv2d_4/biascustom_model_1/conv2d_5/kernelcustom_model_1/conv2d_5/biascustom_model_1/dense_2/kernelcustom_model_1/dense_2/biascustom_model_1/dense_3/kernelcustom_model_1/dense_3/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_save_113065926
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecustom_model_1/conv2d_3/kernelcustom_model_1/conv2d_3/biascustom_model_1/conv2d_4/kernelcustom_model_1/conv2d_4/biascustom_model_1/conv2d_5/kernelcustom_model_1/conv2d_5/biascustom_model_1/dense_2/kernelcustom_model_1/dense_2/biascustom_model_1/dense_3/kernelcustom_model_1/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference__traced_restore_113065965��
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065599

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_conv2d_5_layer_call_fn_113065783

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065588w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs:)%
#
_user_specified_name	113065777:)%
#
_user_specified_name	113065779
�
�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065774

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������		@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
,__inference_conv2d_3_layer_call_fn_113065743

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065556w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������TT: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs:)%
#
_user_specified_name	113065737:)%
#
_user_specified_name	113065739
�#
�
M__inference_custom_model_1_layer_call_and_return_conditional_losses_113065633
input_1,
conv2d_3_113065557:  
conv2d_3_113065559: ,
conv2d_4_113065573: @ 
conv2d_4_113065575:@,
conv2d_5_113065589:@@ 
conv2d_5_113065591:@%
dense_2_113065612:
�� 
dense_2_113065614:	�$
dense_3_113065627:	�	
dense_3_113065629:	
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall^
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:���������TT�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_3_113065557conv2d_3_113065559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065556�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_113065573conv2d_4_113065575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065572�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_113065589conv2d_5_113065591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065588�
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065599�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_113065612dense_2_113065614*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_113065611�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_113065627dense_3_113065629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_113065626w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������TT: : : : : : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1:)%
#
_user_specified_name	113065557:)%
#
_user_specified_name	113065559:)%
#
_user_specified_name	113065573:)%
#
_user_specified_name	113065575:)%
#
_user_specified_name	113065589:)%
#
_user_specified_name	113065591:)%
#
_user_specified_name	113065612:)%
#
_user_specified_name	113065614:)	%
#
_user_specified_name	113065627:)
%
#
_user_specified_name	113065629
�
�
,__inference_conv2d_4_layer_call_fn_113065763

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065572w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������		@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:)%
#
_user_specified_name	113065757:)%
#
_user_specified_name	113065759
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065805

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065556

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������TT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
2__inference_custom_model_1_layer_call_fn_113065658
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�	
	unknown_8:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_custom_model_1_layer_call_and_return_conditional_losses_113065633o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������TT: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1:)%
#
_user_specified_name	113065636:)%
#
_user_specified_name	113065638:)%
#
_user_specified_name	113065640:)%
#
_user_specified_name	113065642:)%
#
_user_specified_name	113065644:)%
#
_user_specified_name	113065646:)%
#
_user_specified_name	113065648:)%
#
_user_specified_name	113065650:)	%
#
_user_specified_name	113065652:)
%
#
_user_specified_name	113065654
�
�
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065794

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065572

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������		@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065588

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
+__inference_dense_2_layer_call_fn_113065814

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_113065611p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:)%
#
_user_specified_name	113065808:)%
#
_user_specified_name	113065810
�

�
F__inference_dense_2_layer_call_and_return_conditional_losses_113065611

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
+__inference_dense_3_layer_call_fn_113065834

inputs
unknown:	�	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_113065626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:)%
#
_user_specified_name	113065828:)%
#
_user_specified_name	113065830
�
�
'__inference_signature_wrapper_113065734
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�	
	unknown_8:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__wrapped_model_113065542o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������TT: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1:)%
#
_user_specified_name	113065712:)%
#
_user_specified_name	113065714:)%
#
_user_specified_name	113065716:)%
#
_user_specified_name	113065718:)%
#
_user_specified_name	113065720:)%
#
_user_specified_name	113065722:)%
#
_user_specified_name	113065724:)%
#
_user_specified_name	113065726:)	%
#
_user_specified_name	113065728:)
%
#
_user_specified_name	113065730
�@
�

$__inference__wrapped_model_113065542
input_1P
6custom_model_1_conv2d_3_conv2d_readvariableop_resource: E
7custom_model_1_conv2d_3_biasadd_readvariableop_resource: P
6custom_model_1_conv2d_4_conv2d_readvariableop_resource: @E
7custom_model_1_conv2d_4_biasadd_readvariableop_resource:@P
6custom_model_1_conv2d_5_conv2d_readvariableop_resource:@@E
7custom_model_1_conv2d_5_biasadd_readvariableop_resource:@I
5custom_model_1_dense_2_matmul_readvariableop_resource:
��E
6custom_model_1_dense_2_biasadd_readvariableop_resource:	�H
5custom_model_1_dense_3_matmul_readvariableop_resource:	�	D
6custom_model_1_dense_3_biasadd_readvariableop_resource:	
identity��.custom_model_1/conv2d_3/BiasAdd/ReadVariableOp�-custom_model_1/conv2d_3/Conv2D/ReadVariableOp�.custom_model_1/conv2d_4/BiasAdd/ReadVariableOp�-custom_model_1/conv2d_4/Conv2D/ReadVariableOp�.custom_model_1/conv2d_5/BiasAdd/ReadVariableOp�-custom_model_1/conv2d_5/Conv2D/ReadVariableOp�-custom_model_1/dense_2/BiasAdd/ReadVariableOp�,custom_model_1/dense_2/MatMul/ReadVariableOp�-custom_model_1/dense_3/BiasAdd/ReadVariableOp�,custom_model_1/dense_3/MatMul/ReadVariableOpm
custom_model_1/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:���������TT�
-custom_model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp6custom_model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
custom_model_1/conv2d_3/Conv2DConv2Dcustom_model_1/Cast:y:05custom_model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
.custom_model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
custom_model_1/conv2d_3/BiasAddBiasAdd'custom_model_1/conv2d_3/Conv2D:output:06custom_model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
custom_model_1/conv2d_3/ReluRelu(custom_model_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
-custom_model_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp6custom_model_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
custom_model_1/conv2d_4/Conv2DConv2D*custom_model_1/conv2d_3/Relu:activations:05custom_model_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
�
.custom_model_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
custom_model_1/conv2d_4/BiasAddBiasAdd'custom_model_1/conv2d_4/Conv2D:output:06custom_model_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@�
custom_model_1/conv2d_4/ReluRelu(custom_model_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@�
-custom_model_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp6custom_model_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
custom_model_1/conv2d_5/Conv2DConv2D*custom_model_1/conv2d_4/Relu:activations:05custom_model_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
.custom_model_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
custom_model_1/conv2d_5/BiasAddBiasAdd'custom_model_1/conv2d_5/Conv2D:output:06custom_model_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
custom_model_1/conv2d_5/ReluRelu(custom_model_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������@o
custom_model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
 custom_model_1/flatten_1/ReshapeReshape*custom_model_1/conv2d_5/Relu:activations:0'custom_model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
,custom_model_1/dense_2/MatMul/ReadVariableOpReadVariableOp5custom_model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
custom_model_1/dense_2/MatMulMatMul)custom_model_1/flatten_1/Reshape:output:04custom_model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-custom_model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp6custom_model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
custom_model_1/dense_2/BiasAddBiasAdd'custom_model_1/dense_2/MatMul:product:05custom_model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
custom_model_1/dense_2/ReluRelu'custom_model_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,custom_model_1/dense_3/MatMul/ReadVariableOpReadVariableOp5custom_model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0�
custom_model_1/dense_3/MatMulMatMul)custom_model_1/dense_2/Relu:activations:04custom_model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
-custom_model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp6custom_model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
custom_model_1/dense_3/BiasAddBiasAdd'custom_model_1/dense_3/MatMul:product:05custom_model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	v
IdentityIdentity'custom_model_1/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp/^custom_model_1/conv2d_3/BiasAdd/ReadVariableOp.^custom_model_1/conv2d_3/Conv2D/ReadVariableOp/^custom_model_1/conv2d_4/BiasAdd/ReadVariableOp.^custom_model_1/conv2d_4/Conv2D/ReadVariableOp/^custom_model_1/conv2d_5/BiasAdd/ReadVariableOp.^custom_model_1/conv2d_5/Conv2D/ReadVariableOp.^custom_model_1/dense_2/BiasAdd/ReadVariableOp-^custom_model_1/dense_2/MatMul/ReadVariableOp.^custom_model_1/dense_3/BiasAdd/ReadVariableOp-^custom_model_1/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������TT: : : : : : : : : : 2`
.custom_model_1/conv2d_3/BiasAdd/ReadVariableOp.custom_model_1/conv2d_3/BiasAdd/ReadVariableOp2^
-custom_model_1/conv2d_3/Conv2D/ReadVariableOp-custom_model_1/conv2d_3/Conv2D/ReadVariableOp2`
.custom_model_1/conv2d_4/BiasAdd/ReadVariableOp.custom_model_1/conv2d_4/BiasAdd/ReadVariableOp2^
-custom_model_1/conv2d_4/Conv2D/ReadVariableOp-custom_model_1/conv2d_4/Conv2D/ReadVariableOp2`
.custom_model_1/conv2d_5/BiasAdd/ReadVariableOp.custom_model_1/conv2d_5/BiasAdd/ReadVariableOp2^
-custom_model_1/conv2d_5/Conv2D/ReadVariableOp-custom_model_1/conv2d_5/Conv2D/ReadVariableOp2^
-custom_model_1/dense_2/BiasAdd/ReadVariableOp-custom_model_1/dense_2/BiasAdd/ReadVariableOp2\
,custom_model_1/dense_2/MatMul/ReadVariableOp,custom_model_1/dense_2/MatMul/ReadVariableOp2^
-custom_model_1/dense_3/BiasAdd/ReadVariableOp-custom_model_1/dense_3/BiasAdd/ReadVariableOp2\
,custom_model_1/dense_3/MatMul/ReadVariableOp,custom_model_1/dense_3/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�5
�
%__inference__traced_restore_113065965
file_prefixI
/assignvariableop_custom_model_1_conv2d_3_kernel: =
/assignvariableop_1_custom_model_1_conv2d_3_bias: K
1assignvariableop_2_custom_model_1_conv2d_4_kernel: @=
/assignvariableop_3_custom_model_1_conv2d_4_bias:@K
1assignvariableop_4_custom_model_1_conv2d_5_kernel:@@=
/assignvariableop_5_custom_model_1_conv2d_5_bias:@D
0assignvariableop_6_custom_model_1_dense_2_kernel:
��=
.assignvariableop_7_custom_model_1_dense_2_bias:	�C
0assignvariableop_8_custom_model_1_dense_3_kernel:	�	<
.assignvariableop_9_custom_model_1_dense_3_bias:	
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp/assignvariableop_custom_model_1_conv2d_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp/assignvariableop_1_custom_model_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp1assignvariableop_2_custom_model_1_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_custom_model_1_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp1assignvariableop_4_custom_model_1_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp/assignvariableop_5_custom_model_1_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_custom_model_1_dense_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_custom_model_1_dense_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_custom_model_1_dense_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_custom_model_1_dense_3_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_11Identity_11:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:>:
8
_user_specified_name custom_model_1/conv2d_3/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_3/bias:>:
8
_user_specified_name custom_model_1/conv2d_4/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_4/bias:>:
8
_user_specified_name custom_model_1/conv2d_5/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_5/bias:=9
7
_user_specified_namecustom_model_1/dense_2/kernel:;7
5
_user_specified_namecustom_model_1/dense_2/bias:=	9
7
_user_specified_namecustom_model_1/dense_3/kernel:;
7
5
_user_specified_namecustom_model_1/dense_3/bias
�\
�

"__inference__traced_save_113065926
file_prefixO
5read_disablecopyonread_custom_model_1_conv2d_3_kernel: C
5read_1_disablecopyonread_custom_model_1_conv2d_3_bias: Q
7read_2_disablecopyonread_custom_model_1_conv2d_4_kernel: @C
5read_3_disablecopyonread_custom_model_1_conv2d_4_bias:@Q
7read_4_disablecopyonread_custom_model_1_conv2d_5_kernel:@@C
5read_5_disablecopyonread_custom_model_1_conv2d_5_bias:@J
6read_6_disablecopyonread_custom_model_1_dense_2_kernel:
��C
4read_7_disablecopyonread_custom_model_1_dense_2_bias:	�I
6read_8_disablecopyonread_custom_model_1_dense_3_kernel:	�	B
4read_9_disablecopyonread_custom_model_1_dense_3_bias:	
savev2_const
identity_21��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead5read_disablecopyonread_custom_model_1_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp5read_disablecopyonread_custom_model_1_conv2d_3_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead5read_1_disablecopyonread_custom_model_1_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp5read_1_disablecopyonread_custom_model_1_conv2d_3_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead7read_2_disablecopyonread_custom_model_1_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp7read_2_disablecopyonread_custom_model_1_conv2d_4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_3/DisableCopyOnReadDisableCopyOnRead5read_3_disablecopyonread_custom_model_1_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp5read_3_disablecopyonread_custom_model_1_conv2d_4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnRead7read_4_disablecopyonread_custom_model_1_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp7read_4_disablecopyonread_custom_model_1_conv2d_5_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_5/DisableCopyOnReadDisableCopyOnRead5read_5_disablecopyonread_custom_model_1_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp5read_5_disablecopyonread_custom_model_1_conv2d_5_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_6/DisableCopyOnReadDisableCopyOnRead6read_6_disablecopyonread_custom_model_1_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp6read_6_disablecopyonread_custom_model_1_dense_2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_7/DisableCopyOnReadDisableCopyOnRead4read_7_disablecopyonread_custom_model_1_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp4read_7_disablecopyonread_custom_model_1_dense_2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead6read_8_disablecopyonread_custom_model_1_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp6read_8_disablecopyonread_custom_model_1_dense_3_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�	*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�	f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�	�
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_custom_model_1_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_custom_model_1_dense_3_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_20Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_21IdentityIdentity_20:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_21Identity_21:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:>:
8
_user_specified_name custom_model_1/conv2d_3/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_3/bias:>:
8
_user_specified_name custom_model_1/conv2d_4/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_4/bias:>:
8
_user_specified_name custom_model_1/conv2d_5/kernel:<8
6
_user_specified_namecustom_model_1/conv2d_5/bias:=9
7
_user_specified_namecustom_model_1/dense_2/kernel:;7
5
_user_specified_namecustom_model_1/dense_2/bias:=	9
7
_user_specified_namecustom_model_1/dense_3/kernel:;
7
5
_user_specified_namecustom_model_1/dense_3/bias:=9

_output_shapes
: 

_user_specified_nameConst
�	
�
F__inference_dense_3_layer_call_and_return_conditional_losses_113065844

inputs1
matmul_readvariableop_resource:	�	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065754

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������TT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
F__inference_dense_3_layer_call_and_return_conditional_losses_113065626

inputs1
matmul_readvariableop_resource:	�	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
F__inference_dense_2_layer_call_and_return_conditional_losses_113065825

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
I
-__inference_flatten_1_layer_call_fn_113065799

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065599a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������TT<
output_10
StatefulPartitionedCall:0���������	tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
flatten

dense1

dense2

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
2__inference_custom_model_1_layer_call_fn_113065658�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
M__inference_custom_model_1_layer_call_and_return_conditional_losses_113065633�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
$__inference__wrapped_model_113065542input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
 &_jit_compiled_convolution_op"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

kernel
bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
,
Gserving_default"
signature_map
8:6 2custom_model_1/conv2d_3/kernel
*:( 2custom_model_1/conv2d_3/bias
8:6 @2custom_model_1/conv2d_4/kernel
*:(@2custom_model_1/conv2d_4/bias
8:6@@2custom_model_1/conv2d_5/kernel
*:(@2custom_model_1/conv2d_5/bias
1:/
��2custom_model_1/dense_2/kernel
*:(�2custom_model_1/dense_2/bias
0:.	�	2custom_model_1/dense_3/kernel
):'	2custom_model_1/dense_3/bias
 "
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_custom_model_1_layer_call_fn_113065658input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_custom_model_1_layer_call_and_return_conditional_losses_113065633input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
Mtrace_02�
,__inference_conv2d_3_layer_call_fn_113065743�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
�
Ntrace_02�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065754�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_02�
,__inference_conv2d_4_layer_call_fn_113065763�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
�
Utrace_02�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065774�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
[trace_02�
,__inference_conv2d_5_layer_call_fn_113065783�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
�
\trace_02�
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065794�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
-__inference_flatten_1_layer_call_fn_113065799�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
�
ctrace_02�
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065805�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
+__inference_dense_2_layer_call_fn_113065814�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
�
jtrace_02�
F__inference_dense_2_layer_call_and_return_conditional_losses_113065825�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
+__inference_dense_3_layer_call_fn_113065834�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
F__inference_dense_3_layer_call_and_return_conditional_losses_113065844�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
�B�
'__inference_signature_wrapper_113065734input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv2d_3_layer_call_fn_113065743inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065754inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv2d_4_layer_call_fn_113065763inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065774inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv2d_5_layer_call_fn_113065783inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065794inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_flatten_1_layer_call_fn_113065799inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065805inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_2_layer_call_fn_113065814inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_2_layer_call_and_return_conditional_losses_113065825inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_3_layer_call_fn_113065834inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_3_layer_call_and_return_conditional_losses_113065844inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
$__inference__wrapped_model_113065542{
8�5
.�+
)�&
input_1���������TT
� "3�0
.
output_1"�
output_1���������	�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_113065754s7�4
-�*
(�%
inputs���������TT
� "4�1
*�'
tensor_0��������� 
� �
,__inference_conv2d_3_layer_call_fn_113065743h7�4
-�*
(�%
inputs���������TT
� ")�&
unknown��������� �
G__inference_conv2d_4_layer_call_and_return_conditional_losses_113065774s7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������		@
� �
,__inference_conv2d_4_layer_call_fn_113065763h7�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������		@�
G__inference_conv2d_5_layer_call_and_return_conditional_losses_113065794s7�4
-�*
(�%
inputs���������		@
� "4�1
*�'
tensor_0���������@
� �
,__inference_conv2d_5_layer_call_fn_113065783h7�4
-�*
(�%
inputs���������		@
� ")�&
unknown���������@�
M__inference_custom_model_1_layer_call_and_return_conditional_losses_113065633t
8�5
.�+
)�&
input_1���������TT
� ",�)
"�
tensor_0���������	
� �
2__inference_custom_model_1_layer_call_fn_113065658i
8�5
.�+
)�&
input_1���������TT
� "!�
unknown���������	�
F__inference_dense_2_layer_call_and_return_conditional_losses_113065825e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_2_layer_call_fn_113065814Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_3_layer_call_and_return_conditional_losses_113065844d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������	
� �
+__inference_dense_3_layer_call_fn_113065834Y0�-
&�#
!�
inputs����������
� "!�
unknown���������	�
H__inference_flatten_1_layer_call_and_return_conditional_losses_113065805h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
tensor_0����������
� �
-__inference_flatten_1_layer_call_fn_113065799]7�4
-�*
(�%
inputs���������@
� ""�
unknown�����������
'__inference_signature_wrapper_113065734�
C�@
� 
9�6
4
input_1)�&
input_1���������TT"3�0
.
output_1"�
output_1���������	