import os
from core.config import Config
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
settings = Config()

def normalize_inputs_train(inputs):
    normalized_inputs = {'Trgt Physically attacked': inputs['Trgt Physically attacked']}
    for key, value in inputs.items():
        if key == 'Trgt Physically attacked':
            continue
        normalized_inputs[f'norm_{key}'] = tft.scale_to_0_1(value)
    return normalized_inputs


def normalize_inputs_test(inputs):
    transform_fn_dir = os.path.join(settings.work_dir, transform_fn_io.TRANSFORM_FN_DIR)
    inputs_copy = inputs.copy()
    transform_fn = tft.TFTransformOutput(transform_output_dir = transform_fn_dir)
    normalized_inputs = {'Trgt Physically attacked': inputs_copy['Trgt Physically attacked']}
    raw_features = {}

    # Copiar valores de entrada a raw_features
    for key, value in inputs_copy.items():
        if key == 'Trgt Physically attacked':
            continue
        raw_features[key] = value

    # Aplicar transformación a raw_features
    transformed_features = transform_fn.transform_raw_features(raw_features)
    for key, value in transformed_features.items():
        normalized_inputs[f'norm_{key}'] = value

    return normalized_inputs


"""def normalize_inputs_train(inputs):
    

    return {
        'Trgt Physically attacked': inputs ['Trgt Physically attacked'],
        'norm_q1 Custom Age': tft.scale_to_0_1 ( inputs ['q1 Custom Age']),
        'norm_q2 Sex': tft.scale_to_0_1 ( inputs ['q2 Sex']),
        'norm_q3 In what grade are you': tft.scale_to_0_1 ( inputs ['q3 In what grade are you']),
        'norm_q4 How often went hungry': tft.scale_to_0_1 ( inputs ['q4 How often went hungry']),
        'norm_q5 Fast food eating ': tft.scale_to_0_1 ( inputs ['q5 Fast food eating ']),
        'norm_q6 Physical fighting': tft.scale_to_0_1 ( inputs ['q6 Physical fighting']),
        'norm_q7 Seriously injured': tft.scale_to_0_1 ( inputs ['q7 Seriously injured']),
        'norm_q8 Serious injury type ': tft.scale_to_0_1 ( inputs ['q8 Serious injury type ']),
        'norm_q9 Serious injury cause': tft.scale_to_0_1 ( inputs ['q9 Serious injury cause']),
        'norm_q10 Felt lonely': tft.scale_to_0_1 ( inputs ['q10 Felt lonely']),
        'norm_q11 Could not sleep': tft.scale_to_0_1 ( inputs ['q11 Could not sleep']),
        'norm_q12 Considered suicide': tft.scale_to_0_1 ( inputs ['q12 Considered suicide']), 
        'norm_q13 Made a suicide plan': tft.scale_to_0_1 ( inputs ['q13 Made a suicide plan']),
        'norm_q14 Attempted suicide': tft.scale_to_0_1 ( inputs ['q14 Attempted suicide']), 
        'norm_q15 Close friends': tft.scale_to_0_1 ( inputs ['q15 Close friends']),
        'norm_q16 Initiation of cigarette use': tft.scale_to_0_1 ( inputs ['q16 Initiation of cigarette use']),
        'norm_q17 Current cigarette use': tft.scale_to_0_1 ( inputs ['q17 Current cigarette use']),
        'norm_q18 Initiation of alcohol use': tft.scale_to_0_1 ( inputs ['q18 Initiation of alcohol use']),
        'norm_q19 Current alcohol use': tft.scale_to_0_1 ( inputs ['q19 Current alcohol use']),
        'norm_q20 Drank 2+ drinks ': tft.scale_to_0_1 ( inputs ['q20 Drank 2+ drinks ']),
        'norm_q21 Source of alchohol': tft.scale_to_0_1 ( inputs ['q21 Source of alchohol']),
        'norm_q22 Really drunk': tft.scale_to_0_1 ( inputs ['q22 Really drunk']),
        'norm_q23 Trouble from drinking': tft.scale_to_0_1 ( inputs ['q23 Trouble from drinking']),
        'norm_q24 Initiation of drug use': tft.scale_to_0_1 ( inputs ['q24 Initiation of drug use']),
        'norm_q25 Ever marijuana use': tft.scale_to_0_1 ( inputs ['q25 Ever marijuana use']),
        'norm_q26 Current marijuana use ': tft.scale_to_0_1 ( inputs ['q26 Current marijuana use ']),
        'norm_q27 Amphethamine or methamphetamine use': tft.scale_to_0_1 ( inputs ['q27 Amphethamine or methamphetamine use']),
        'norm_q28 Ever sexual intercourse': tft.scale_to_0_1 ( inputs ['q28 Ever sexual intercourse']),
        'norm_q29 Age first had sex': tft.scale_to_0_1 ( inputs ['q29 Age first had sex']),
        'norm_q30 Number of sex partners': tft.scale_to_0_1 ( inputs ['q30 Number of sex partners']),
        'norm_q31 Condom use': tft.scale_to_0_1 ( inputs ['q31 Condom use']),
        'norm_q32 Birth control used ': tft.scale_to_0_1 ( inputs ['q32 Birth control used ']), 
        'norm_q33 Physical activity past 7 days': tft.scale_to_0_1 ( inputs ['q33 Physical activity past 7 days']),
        'norm_q34 Walk or bike to school': tft.scale_to_0_1 ( inputs ['q34 Walk or bike to school']),
        'norm_q35 PE attendance': tft.scale_to_0_1 ( inputs ['q35 PE attendance']),
        'norm_q36 Sitting activities': tft.scale_to_0_1 ( inputs ['q36 Sitting activities']),
        'norm_q37 Miss school no permission': tft.scale_to_0_1 ( inputs ['q37 Miss school no permission']),
        'norm_q38 Other students kind and helpful ': tft.scale_to_0_1 ( inputs ['q38 Other students kind and helpful ']),
        'norm_q39 Parents check homework ': tft.scale_to_0_1 ( inputs ['q39 Parents check homework ']),
        'norm_q40 Parents understand problems': tft.scale_to_0_1 ( inputs ['q40 Parents understand problems']),
        'norm_q41 Parents know about free time': tft.scale_to_0_1 ( inputs ['q41 Parents know about free time']),
        'norm_q42 Parents go through their things': tft.scale_to_0_1 ( inputs ['q42 Parents go through their things'])        
    }"""

"""def normalize_inputs(inputs, mode):
    transform_fn_dir = os.path.join(settings.work_dir, transform_fn_io.TRANSFORM_FN_DIR)
    def scale_to_0_1(inputs):
        # Define the normalization parameters
        feature_range_min = tf.constant(0.0, dtype=tf.float32)
        feature_range_max = tf.constant(1.0, dtype=tf.float32)

        # Scale the inputs
        inputs_scaled, _ = tft.scale_to_0_1(inputs)
        inputs_scaled = tf.cast(inputs_scaled, dtype=tf.float32)

        # Rescale the inputs to the desired feature range
        inputs_rescaled = feature_range_min + \
            inputs_scaled * (feature_range_max - feature_range_min)

        return inputs_rescaled

    # Scale the inputs if it is train mode
    if mode == 'train':
        outputs = scale_to_0_1(inputs)
    # Download the normalization statistics and apply them if it is test mode
    else:
        # Download the statistics from the saved model
        transform_fn = tft.TFTransformOutput(transform_fn_dir)
        statistics = transform_fn.transform_fn.metadata.schema.feature
        # Apply the scaling transformation using the downloaded statistics
        outputs, _ = tft.scale_to_0_1(inputs, elementwise=True, 
                                      desired_outputs=None, 
                                      output_range=(0.0, 1.0), 
                                      stats=statistics['input_name'].scaling)
        outputs = tf.cast(outputs, dtype=tf.float32)

    return outputs"""