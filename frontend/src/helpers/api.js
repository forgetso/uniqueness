var getRelationQuery = function () {
    return fetch(process.env.REACT_APP_API_URL + 'relationquery',
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

var saveResponse = function (cue, relations, relation_type, time_duration, fingerprint) {
    return fetch(process.env.REACT_APP_API_URL + "relationquery",
        {
            method: 'POST',
            body: JSON.stringify({
                'cue': cue,
                'relation_type': relation_type,
                'relations': relations,
                'time_duration': time_duration,
                'fingerprint': fingerprint
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

exports.getRelationQuery = getRelationQuery;
exports.saveResponse = saveResponse;
