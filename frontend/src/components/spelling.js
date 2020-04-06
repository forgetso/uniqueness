import React, {Component} from 'react';
import App from "../App";

class Spelling extends Component {
    constructor(props) {
        super(props);
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
        <li key={correction.toString()}>{correction}</li>
    );
    return (
        <ul>{listItems}</ul>
    );
}

export default Spelling;