var getCueWord = function () {
    return fetch(process.env.REACT_APP_API_URL + 'cueresponse',
        {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }).then((res) => {
        return res.json();
    }).then((data) => {
        return data;
    });
};

var saveResponseWord = function (cue, response, time_duration, user_id) {
    return fetch(process.env.REACT_APP_API_URL + "cueresponse",
        {
            method: 'POST',
            body: JSON.stringify({
                'cue': cue,
                'response': response,
                'time_duration': time_duration,
                'user_id': user_id
            }),
            headers: {
                //'X-Api-Key': API_KEY,
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }
    ).then(function (response) {
        return response
    });
};

var correctSpelling = function (response_id, original_word, correction) {
    return fetch(process.env.REACT_APP_API_URL + "correctspelling",
        {
            method: 'POST',
            body: JSON.stringify({
                'response_id': response_id,
                'original_word': original_word,
                'correction': correction
            }),
            headers: {
                //'X-Api-Key': API_KEY,
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }

        }).then(function (response) {
        return response
    });
};

exports.getCueWord = getCueWord;
exports.saveResponseWord = saveResponseWord;
exports.correctSpelling = correctSpelling;