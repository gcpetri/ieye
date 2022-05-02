from flask import request, send_file, Blueprint, jsonify
from werkzeug.utils import secure_filename
from app.Services.prepare import prepare_image, test_prepare_image
from app.Services.test import get_test_image
from app.Services.harris import flann_match, sift_detect
from app.Services.zip_images import encode_images
from app.Services.db import eye_collection, insert_eye

main_blueprint = Blueprint('main_blueprint', __name__, url_prefix='/')

@main_blueprint.route('/go-test', methods=['GET'])
def test():
    img = get_test_image(1)

    # resize, black and white, smoothed, and contrasted
    prepared_image = test_prepare_image(img)

    # get the keypoints and image
    sift_image, _, _ = sift_detect(prepared_image)

    # show all images that we want
    zip_stream = encode_images([['original', img], ['prepared image', prepared_image], ['sift data', sift_image]])

    return send_file(
        zip_stream,
        attachment_filename='res.zip',
        mimetype='application/x-zip-compressed'
    )

@main_blueprint.route('/upload', methods=['POST'])
def post_image():
    image = request.files['file']
    if not image:
        return jsonify({'reason': 'image not found'}), 400

    print('filename', image.filename)
    file_name = secure_filename(image.filename) # should remove '$' characters
    print('file_name', file_name)

    content_type = request.mimetype
    print('content_type', content_type)

    # they're logging in so try to match them
    if file_name == 'null':
        # resize, black and white, smoothed, and contrasted
        _, prepared_image = prepare_image(image)

        # get the keypoints and image
        sift_image, kp, des = sift_detect(prepared_image)

        # show all matching data
        match_data = flann_match(prepared_image, kp, des)

        # show all images that we want
        formatted_match_data = [["{}_{}".format(match[0], match[1]), match[2]] for match in match_data]
        zip_stream = encode_images(formatted_match_data)

        return send_file(
            zip_stream,
            attachment_filename='res.zip',
            mimetype='application/x-zip-compressed'
        )

    # they're signing up so save their data
    else:

        # check if anyone with that name exists
        if eye_collection.find_one({ "name": file_name }) != None:
            return jsonify({'reason': 'Person with that name already exists. Please enter a new name.'}), 400

        # resize, black and white, smoothed, and contrasted
        original_image, prepared_image = prepare_image(image)

        # get the keypoints and image
        sift_image, kp, des = sift_detect(prepared_image)
    
        # show all images that we want
        zip_stream = encode_images([['original', original_image], ['prepared image', prepared_image], ['sift data', sift_image]])

        # save kp to db
        insert_eye(file_name, prepared_image, kp, des)

        return send_file(
            zip_stream,
            attachment_filename='res.zip',
            mimetype='application/x-zip-compressed'
        )

@main_blueprint.route('/num-users', methods=['GET'])
def get_num_users():
    return jsonify({'num': eye_collection.count_documents({})})
    