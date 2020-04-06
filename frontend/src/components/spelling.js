import React, {Component} from 'react';
import App from "../App";
import * as api from "../helpers/api.js"

class Spelling extends Component {
    constructor(props) {
        super(props);
        console.log(this.props.cue);
        console.log(this.props.response);
        console.log(this.props.corrections);
    }

    render() {
        return (
            <div>
                <div>Did you mean</div>
                <div>
                    <CorrectionsList corrections={this.props.corrections}/>
                </div>
                <div>?</div>
            </div>

        );
    }
}

function CorrectionsList(props) {
    console.log(props);
    const corrections = props.corrections;
    const listItems = corrections.map((correction) =>
        <li key={correction.toString()}><a href=""
                                           onClick={sendSpellingCorrection(correction.toString())}>{correction}</a>
        </li>
    );
    return (
        <ul className="list-group">{listItems}</ul>
    );
}

function sendSpellingCorrection(correction) {
    api.correctSpelling(this.props.responseId,  correction)
}

export default Spelling;