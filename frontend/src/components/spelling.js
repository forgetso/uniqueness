'use strict'
import React, {Component} from 'react';
import CorrectionsList from "./correctionslist"

class Spelling extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div>
                <div>Did you mean</div>
                <div>
                    <CorrectionsList response_id={this.props.response_id}
                                     corrections={this.props.corrections}
                                     showSpelling={this.props.showSpelling}
                                     sendSpellingCorrection={this.props.sendSpellingCorrection}
                                     reload={this.props.reload}
                    />
                </div>
                <div>?</div>
            </div>

        );
    }
}


export default Spelling;